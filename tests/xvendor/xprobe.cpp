#include "gpu_runtime.hpp"
#include "xprobe_api.hpp"
#include <cstdint>
#include <exception>
#include <vector>

namespace {
void* host = nullptr;
void* device = nullptr;
size_t bytes = 0;
std::string last_error;
bool initialized = false;
template<class F> int guarded(F function) {
    try { function(); last_error.clear(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return 1; }
}
int init() { return guarded([] {
    if (initialized) throw std::runtime_error("probe already initialized");
    xvendor::gpu_init(); initialized = true;
}); }
int register_host(void* pointer, size_t size) { return guarded([&] {
    if (!initialized || host || !pointer || size < 4096 || size % 4096)
        throw std::runtime_error("invalid probe registration state/size");
    XV_GPU(cudaHostRegister(pointer, size, xvendor::registration_flags()));
    host = pointer; bytes = size;
    XV_GPU(cudaMalloc(&device, size));
}); }
int check() { return guarded([] {
    if (!host || !device) throw std::runtime_error("probe has no registered buffer");
    auto* words = static_cast<uint32_t*>(host);
    for (size_t i = 0; i < bytes / sizeof(uint32_t); ++i) words[i] = static_cast<uint32_t>(i * 17 + 3);
    XV_GPU(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    std::vector<uint32_t> returned(bytes / sizeof(uint32_t));
    XV_GPU(cudaMemcpy(returned.data(), device, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < returned.size(); ++i)
        if (returned[i] != words[i]) throw std::runtime_error("probe H2D/D2H mismatch");
}); }
int unregister_host() { return guarded([] {
    if (device) { XV_GPU(cudaFree(device)); device = nullptr; }
    if (host) { XV_GPU(cudaHostUnregister(host)); host = nullptr; bytes = 0; }
}); }
int shutdown() { return guarded([] {
    if (host || device) throw std::runtime_error("unregister before shutdown");
    if (initialized) { xvendor::gpu_reset(); initialized = false; }
}); }
const char* error() { return last_error.c_str(); }
const XProbeAPI api{1, init, register_host, check, unregister_host, shutdown, error};
}
extern "C" __attribute__((visibility("default"))) const XProbeAPI* strata_xprobe_v1() { return &api; }
