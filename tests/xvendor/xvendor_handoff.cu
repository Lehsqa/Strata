// Manual Linux feasibility probe, deliberately separate from the engine.
#include "gpu_runtime.hpp"
#include "host_pingpong.hpp"
#include "xvendor_cli.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <sys/resource.h>
#include <sys/statvfs.h>

namespace {
using Clock = std::chrono::steady_clock;

__global__ void delay_kernel(unsigned long long ticks) {
    const unsigned long long begin = clock64();
    while (clock64() - begin < ticks) {}
}
__global__ void validate_and_return(const float* outbound, float* returned, int32_t* indices,
                                   const uint32_t* expected, int32_t* errors) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t sequence = *reinterpret_cast<const volatile uint32_t*>(expected);
    if (i < xvendor::kOutboundFloats && outbound[i] != static_cast<float>(sequence + i % 13)) atomicExch(errors, 1);
    if (i < xvendor::kReturnFloats) returned[i] = static_cast<float>(sequence + i % 17);
    if (i < xvendor::kRows) indices[i] = static_cast<int32_t>(sequence + i);
}

struct Registered {
    void* host;
    void* mapped = nullptr;
    explicit Registered(xvendor::Region& region) : host(region.data()) {
        XV_GPU(cudaHostRegister(host, region.size(), xvendor::registration_flags()));
        try { XV_GPU(cudaHostGetDevicePointer(&mapped, host, 0)); }
        catch (...) { cudaHostUnregister(host); throw; }
    }
    void release() { XV_GPU(cudaHostUnregister(host)); host = nullptr; }
    ~Registered() { if (!std::uncaught_exceptions() && host) cudaHostUnregister(host); }
};
struct Arena {
    void* pointer = MAP_FAILED;
    size_t bytes;
    explicit Arena(size_t size) : bytes(size) {
        pointer = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pointer == MAP_FAILED) throw xvendor::system_error("mmap anonymous arena");
        // Volatile page stores commit private RAM in each process; a sparse mapping is not this test.
        auto* pages = static_cast<volatile unsigned char*>(pointer);
        const size_t page = xvendor::page_size();
        for (size_t i = 0; i < bytes; i += page) pages[i] = 1;
    }
    ~Arena() { if (pointer != MAP_FAILED) munmap(pointer, bytes); }
};
struct DeviceMemory {
    void* pointer = nullptr;
    explicit DeviceMemory(size_t bytes) { XV_GPU(cudaMalloc(&pointer, bytes)); }
    ~DeviceMemory() { if (!std::uncaught_exceptions() && pointer) cudaFree(pointer); }
    template<class T> T* as() const { return static_cast<T*>(pointer); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    template<class F> void capture(cudaStream_t stream, F function) {
        XV_GPU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        function();
        XV_GPU(cudaStreamEndCapture(stream, &graph));
        XV_GPU(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    void launch(cudaStream_t stream) const { XV_GPU(cudaGraphLaunch(exec, stream)); }
    ~Graph() {
        if (std::uncaught_exceptions()) return;
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
};
void wait(xvendor::Sequence& sequence, uint32_t value, const xvendor::WaitOptions& options) {
    xvendor::require_ready(xvendor::wait_sequence(sequence, value, options));
}
void barrier(xvendor::Region& region, bool main, uint32_t sequence, const xvendor::WaitOptions& options) {
    auto& header = region.header();
    if (main) {
        xvendor::publish(header.request, sequence);
        wait(header.response, sequence, options);
    } else {
        wait(header.request, sequence, options);
        xvendor::publish(header.response, sequence);
    }
}
void limits() {
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit)) throw xvendor::system_error("getrlimit MEMLOCK");
    std::printf("RLIMIT_MEMLOCK soft=%llu hard=%llu (RLIM_INFINITY=%llu)\n",
        static_cast<unsigned long long>(limit.rlim_cur), static_cast<unsigned long long>(limit.rlim_max),
        static_cast<unsigned long long>(RLIM_INFINITY));
    struct statvfs info{};
    if (statvfs("/dev/shm", &info)) throw xvendor::system_error("statvfs /dev/shm");
    std::printf("/dev/shm total=%llu available=%llu bytes; memfd is not allocated on this mount\n",
        static_cast<unsigned long long>(info.f_blocks) * info.f_frsize,
        static_cast<unsigned long long>(info.f_bavail) * info.f_frsize);
    std::fflush(stdout);
}
void registration(xvendor::Region& region, const xvendor::Arguments& args, const xvendor::WaitOptions& options) {
    limits();
    // Both processes register the same memfd, then commit independent anonymous arenas.
    Registered registered(region);
    Arena arena(xvendor::checked_mul(args.arena_mib, size_t{1024 * 1024}));
    const bool main = args.role == "main";
    barrier(region, main, 1, options);
    std::printf("simultaneously registered=%zu bytes private committed arena=%zu bytes per process\n", region.size(), arena.bytes);
    const size_t chunk = std::min(region.size() - xvendor::page_size(), size_t{64} * 1024 * 1024);
    DeviceMemory device(chunk);
    // Separate payload from barrier header; both processes read identical registered pages.
    auto* source = static_cast<unsigned char*>(region.data()) + xvendor::page_size();
    if (main) {
        const size_t page = xvendor::page_size();
        for (size_t i = page; i < region.size(); i += page)
            static_cast<unsigned char*>(region.data())[i] = 0x5a;
        std::memset(source, 0x5a, chunk);
    }
    barrier(region, main, 2, options);
    const auto begin = Clock::now();
    constexpr unsigned repeats = 16;
    for (unsigned i = 0; i < repeats; ++i) XV_GPU(cudaMemcpy(device.pointer, source, chunk, cudaMemcpyHostToDevice));
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    std::vector<unsigned char> returned(chunk);
    XV_GPU(cudaMemcpy(returned.data(), device.pointer, chunk, cudaMemcpyDeviceToHost));
    if (!std::all_of(returned.begin(), returned.end(), [](unsigned char byte) { return byte == 0x5a; }))
        throw std::runtime_error("registered H2D/D2H payload mismatch");
    std::printf("registered H2D chunk=%zu repeats=%u bandwidth=%.3f GB/s (synchronous wall time)\n",
        chunk, repeats, static_cast<double>(chunk) * repeats / seconds / 1e9);
    barrier(region, main, 3, options);
    registered.release();
    std::puts("PASS: simultaneous large registration + committed private arenas + registered transfers");
}
void handoff(xvendor::Region& region, const xvendor::Arguments& args, xvendor::WaitOptions options) {
    const bool main = args.role == "main", doorbell = args.mode == "doorbell";
    const auto layout = xvendor::Layout::standard();
    Registered registered(region);
    auto* alias = static_cast<unsigned char*>(registered.mapped);
    auto* mapped_header = reinterpret_cast<xvendor::SharedHeader*>(alias);
    auto* mapped_outbound = reinterpret_cast<float*>(alias + layout.outbound_offset);
    auto* mapped_returned = reinterpret_cast<float*>(alias + layout.return_offset);
    auto* mapped_indices = reinterpret_cast<int32_t*>(alias + layout.indices_offset);
    DeviceMemory outbound(xvendor::kOutboundFloats * sizeof(float));
    DeviceMemory returned(xvendor::kReturnFloats * sizeof(float));
    DeviceMemory indices(xvendor::kRows * sizeof(int32_t));
    DeviceMemory errors(sizeof(int32_t));
    cudaStream_t stream = nullptr;
    XV_GPU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto& header = region.header();
    if (header.gpu_request.load() || header.gpu_response.load())
        throw std::runtime_error("handoff requires a fresh zeroed memfd");
    barrier(region, main, 1, options);
    // Payload round r uses host words offset by one so startup cannot satisfy a round wait.
    cudaDeviceProp properties{};
    XV_GPU(cudaGetDeviceProperties(&properties, 0));
    Graph publish_graph, receive_graph, worker_graph;
    if (main) {
        publish_graph.capture(stream, [&] {
            if (doorbell && args.delay_us) delay_kernel<<<1, 1, 0, stream>>>(
                static_cast<unsigned long long>(args.delay_us) * properties.clockRate / 1000);
            strata::kernels::copy_from_mapped(mapped_outbound, outbound.as<float>(), xvendor::kOutboundFloats, reinterpret_cast<void*>(stream));
            strata::kernels::doorbell_ring(&mapped_header->gpu_request.value, reinterpret_cast<void*>(stream));
        });
        receive_graph.capture(stream, [&] {
            if (doorbell) strata::kernels::doorbell_wait(&mapped_header->gpu_response.value, &mapped_header->gpu_expected, reinterpret_cast<void*>(stream));
            strata::kernels::copy_from_mapped(returned.as<float>(), mapped_returned, xvendor::kReturnFloats, reinterpret_cast<void*>(stream));
            strata::kernels::copy_i32_from_mapped(indices.as<int32_t>(), mapped_indices, xvendor::kRows, reinterpret_cast<void*>(stream));
        });
    } else {
        worker_graph.capture(stream, [&] {
            if (doorbell) strata::kernels::doorbell_wait(&mapped_header->gpu_request.value, &mapped_header->gpu_expected, reinterpret_cast<void*>(stream));
            strata::kernels::copy_from_mapped(outbound.as<float>(), mapped_outbound, xvendor::kOutboundFloats, reinterpret_cast<void*>(stream));
            validate_and_return<<<static_cast<unsigned>((xvendor::kOutboundFloats + 255) / 256), 256, 0, stream>>>(
                outbound.as<float>(), returned.as<float>(), indices.as<int32_t>(), &mapped_header->gpu_expected, errors.as<int32_t>());
            strata::kernels::copy_from_mapped(mapped_returned, returned.as<float>(), xvendor::kReturnFloats, reinterpret_cast<void*>(stream));
            strata::kernels::copy_i32_from_mapped(mapped_indices, indices.as<int32_t>(), xvendor::kRows, reinterpret_cast<void*>(stream));
            strata::kernels::doorbell_ring(&mapped_header->gpu_response.value, reinterpret_cast<void*>(stream));
        });
    }
    XV_GPU(cudaGetLastError());
    auto gpu_poll = options;
    gpu_poll.use_futex = false; // GPU stores cannot issue a Linux futex wake.
    std::vector<float> source(xvendor::kOutboundFloats), result(xvendor::kReturnFloats);
    std::vector<int32_t> ids(xvendor::kRows);
    std::vector<double> times;
    if (main) times.reserve(args.rounds);
    for (uint32_t round = 1; round <= args.rounds; ++round) {
        const uint32_t launch = 2 * round + 1, finish = launch + 1;
        if (main) {
            __atomic_store_n(&header.gpu_expected, round, __ATOMIC_RELEASE);
            for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(round + i % 13);
            XV_GPU(cudaMemcpyAsync(outbound.pointer, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            XV_GPU(cudaStreamSynchronize(stream));
            const auto begin = Clock::now();
            if (doorbell) {
                xvendor::publish(header.request, launch);
                wait(header.response, launch, options); // Worker wait graph is queued before delayed producer.
                publish_graph.launch(stream);
                receive_graph.launch(stream);
                wait(header.gpu_response, round, gpu_poll);
                // Observe unsynchronized GPU publication before any driver query/sync on either producer.
                for (size_t i = 0; i < xvendor::kReturnFloats; ++i)
                    if (region.returned()[i] != static_cast<float>(round + i % 17)) throw std::runtime_error("stale unsynchronized GPU return");
                for (size_t i = 0; i < xvendor::kRows; ++i)
                    if (region.indices()[i] != static_cast<int32_t>(round + i)) throw std::runtime_error("stale unsynchronized GPU indices");
                xvendor::publish(header.request, finish);
                wait(header.response, finish, options);
            } else {
                publish_graph.launch(stream);
                XV_GPU(cudaStreamSynchronize(stream));
                xvendor::publish(header.request, launch, options.use_futex);
                wait(header.response, finish, options);
                receive_graph.launch(stream);
            }
            XV_GPU(cudaStreamSynchronize(stream));
            times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
            XV_GPU(cudaMemcpy(result.data(), returned.pointer, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
            XV_GPU(cudaMemcpy(ids.data(), indices.pointer, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < result.size(); ++i)
                if (result[i] != static_cast<float>(round + i % 17)) throw std::runtime_error("stale GPU return copy");
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] != static_cast<int32_t>(round + i)) throw std::runtime_error("stale GPU indices copy");
        } else {
            wait(header.request, launch, options);
            XV_GPU(cudaMemsetAsync(errors.pointer, 0, sizeof(int32_t), stream));
            worker_graph.launch(stream);
            if (doorbell) {
                xvendor::publish(header.response, launch);
                wait(header.request, finish, options); // Host observed the ring and payload before producer sync.
            }
            XV_GPU(cudaStreamSynchronize(stream));
            int32_t error = 0;
            XV_GPU(cudaMemcpy(&error, errors.pointer, sizeof(error), cudaMemcpyDeviceToHost));
            if (error) throw std::runtime_error("stale GPU outbound input");
            xvendor::publish(header.response, finish, options.use_futex);
        }
    }
    XV_GPU(cudaStreamDestroy(stream));
    registered.release();
    if (main) {
        std::sort(times.begin(), times.end());
        const double p50 = times[(times.size() - 1) * 50 / 100], p99 = times[(times.size() - 1) * 99 / 100];
        std::printf("PASS: mode=%s rounds=%u zero stale values outbound=8x12804 f32 return=8x10240 f32 indices=8 i32 graph replay\n", args.mode.c_str(), args.rounds);
        std::printf("p50_us=%.3f p99_us=%.3f p99_lt_200=%s\n", p50, p99, p99 < 200 ? "yes" : "no");
        if (doorbell) std::printf("extent: includes delayed producer (%u us), worker enqueue/check, host ring/payload observation and acknowledgment; later pipeline gate only\n", args.delay_us);
        else std::puts("extent: excludes main host fill/upload and return check/readback; includes both graph directions, CPU rendezvous and full worker processing/check");
        if (args.rounds < 10000) std::puts("SMOKE ONLY: fewer than 10000 rounds cannot clear the feasibility gate");
    }
}
}
int main(int argc, char** argv) {
    try {
        const auto args = xvendor::arguments(argc, argv);
        if (args.region_bytes) { std::printf("%zu\n", args.bytes()); return 0; }
        auto region = xvendor::Region::map(args.fd, args.bytes());
        xvendor::WaitOptions options;
        options.use_futex = args.mode != "spin";
        options.timeout_ms = args.timeout_ms;
        try {
        xvendor::gpu_init();
        if (args.mode == "large-registration") registration(region, args, options);
        else handoff(region, args, options);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s (registration failures may require a higher RLIMIT_MEMLOCK)\n", error.what());
            std::fflush(nullptr);
            // Exit with the mapping still alive: an outstanding GPU wait cannot be safely unwound.
            std::_Exit(1);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s (registration failures may require a higher RLIMIT_MEMLOCK)\n", error.what());
        std::fflush(nullptr);
        // A failed device wait can make runtime teardown hang. The launcher kills/reaps both peers.
        std::_Exit(1);
    }
}
