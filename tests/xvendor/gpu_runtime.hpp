#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace xvendor {
inline void gpu_check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
}
#define XV_GPU(call) ::xvendor::gpu_check((call), #call)
inline void gpu_init() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    XV_GPU(hipSetDeviceFlags(hipDeviceMapHost));
#else
    XV_GPU(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif
    int count = 0;
    XV_GPU(cudaGetDeviceCount(&count));
    if (!count) {
        std::puts("visible_count=0");
        throw std::runtime_error("no visible GPU in this runtime");
    }
    XV_GPU(cudaSetDevice(0));
    cudaDeviceProp prop{};
    XV_GPU(cudaGetDeviceProperties(&prop, 0));
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    std::printf("vendor=HIP visible_count=%d device=0 name=%s arch=%s\n", count, prop.name, prop.gcnArchName);
#else
    std::printf("vendor=CUDA visible_count=%d device=0 name=%s arch=sm_%d%d\n", count, prop.name, prop.major, prop.minor);
#endif
    std::fflush(stdout);
}
inline unsigned registration_flags() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    return cudaHostRegisterMapped;
#else
    return cudaHostRegisterMapped | cudaHostRegisterPortable;
#endif
}
inline void gpu_reset() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    XV_GPU(hipDeviceReset());
#else
    XV_GPU(cudaDeviceReset());
#endif
}
}
