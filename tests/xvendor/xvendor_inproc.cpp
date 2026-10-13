#include "shm_region.hpp"
#include "xprobe_api.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <string>

namespace {
struct Library {
    void* handle = nullptr;
    const XProbeAPI* api = nullptr;
    bool initialized = false;
    ~Library() {
        if (api && initialized) { api->unregister_host(); api->shutdown(); }
        if (handle) dlclose(handle);
    }
    void open(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(std::string("dlopen ") + path + ": " + dlerror());
        dlerror();
        auto entry = reinterpret_cast<XProbeEntry>(dlsym(handle, "strata_xprobe_v1"));
        const char* error = dlerror();
        if (error) throw std::runtime_error(std::string("probe entry: ") + error);
        api = entry();
        if (!api || api->abi_version != 1 || !api->init || !api->register_host || !api->check ||
            !api->unregister_host || !api->shutdown || !api->error)
            throw std::runtime_error("invalid probe ABI");
    }
    void check(int result) const {
        if (result) throw std::runtime_error(api->error());
    }
};
void visibility() {
    for (const char* key : {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"})
        std::printf("%s=%s\n", key, std::getenv(key) ? std::getenv(key) : "<unset>");
}
}
int main(int argc, char** argv) {
    try {
        std::string cuda, hip, order = "cuda-first", teardown = "hip-first";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("options require values");
            const std::string key(argv[i]), value(argv[i + 1]);
            if (key == "--cuda") cuda = value;
            else if (key == "--hip") hip = value;
            else if (key == "--order") order = value;
            else if (key == "--teardown") teardown = value;
            else throw std::invalid_argument("unknown option " + key);
        }
        if (cuda.empty() || hip.empty() || (order != "cuda-first" && order != "hip-first") ||
            (teardown != "cuda-first" && teardown != "hip-first"))
            throw std::invalid_argument("usage: --cuda LIB --hip LIB --order cuda-first|hip-first --teardown cuda-first|hip-first");
        visibility();
        auto region = xvendor::Region::create(xvendor::page_size() * 256);
        std::array<Library, 2> libs;
        const int first = order == "cuda-first" ? 0 : 1;
        const int stop = teardown == "cuda-first" ? 0 : 1;
        const std::array<std::string, 2> paths{cuda, hip};
        for (int index : {first, 1 - first}) {
            auto& lib = libs[index];
            lib.open(paths[index]);
            lib.check(lib.api->init()); lib.initialized = true;
        }
        for (int index : {first, 1 - first}) libs[index].check(libs[index].api->register_host(region.data(), region.size()));
        // Both registrations remain live during each runtime's real transfer check.
        for (int index : {first, 1 - first}) libs[index].check(libs[index].api->check());
        for (int index : {stop, 1 - stop}) {
            auto& lib = libs[index];
            lib.check(lib.api->unregister_host());
            lib.check(lib.api->shutdown()); lib.initialized = false;
            if (dlclose(lib.handle)) throw std::runtime_error(dlerror());
            lib.handle = nullptr; lib.api = nullptr;
            if (index == stop) libs[1 - stop].check(libs[1 - stop].api->check());
        }
        std::printf("PASS: simultaneous registration, H2D/D2H, init=%s teardown=%s dlclose\n", order.c_str(), teardown.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
