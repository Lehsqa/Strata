#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <sys/mman.h>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) {std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1);} } while(0)
constexpr size_t outbound_n=8*12804, returned_n=8*10240, rows=8;
__global__ void validate_and_return(const float* outbound, float* returned, int32_t* indices,
                                   const uint32_t* expected, int32_t* errors) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t sequence = *reinterpret_cast<const volatile uint32_t*>(expected);
    if (i < outbound_n && outbound[i] != static_cast<float>(sequence + i % 13)) atomicExch(errors, 1);
    if (i < returned_n) returned[i] = static_cast<float>(sequence + i % 17);
    if (i < rows) indices[i] = static_cast<int32_t>(sequence + i);
}
int main() {
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
    std::printf("CUDA metadata diagnostic device=%s sm_%d%d threads=%zu; identical kernel/full checking\n",prop.name,prop.major,prop.minor,outbound_n);
    auto* host=static_cast<uint32_t*>(mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    if(host==MAP_FAILED)return 2;
    CHECK(cudaHostRegister(host,4096,cudaHostRegisterPortable|cudaHostRegisterMapped));
    uint32_t* mapped=nullptr;CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host,0));
    uint32_t* expected=nullptr;float* outbound=nullptr;float* returned=nullptr;int32_t* indices=nullptr;int32_t* errors=nullptr;
    CHECK(cudaMalloc(&expected,4));CHECK(cudaMalloc(&outbound,outbound_n*4));CHECK(cudaMalloc(&returned,returned_n*4));CHECK(cudaMalloc(&indices,rows*4));CHECK(cudaMalloc(&errors,4));
    cudaStream_t stream=nullptr;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
    std::vector<float> source(outbound_n),result(returned_n);std::vector<int32_t> ids(rows);
    for(int mode: {0,1,0,1}) {
        std::vector<float> times;
        for(unsigned r=1;r<=55;++r) {
            __atomic_store_n(host,r,__ATOMIC_RELEASE);
            for(size_t i=0;i<source.size();++i)source[i]=static_cast<float>(r+i%13);
            CHECK(cudaMemcpyAsync(outbound,source.data(),source.size()*4,cudaMemcpyHostToDevice,stream));
            CHECK(cudaMemcpyAsync(expected,host,4,cudaMemcpyHostToDevice,stream));
            CHECK(cudaMemsetAsync(errors,0,4,stream));CHECK(cudaStreamSynchronize(stream));
            CHECK(cudaEventRecord(begin,stream));
            validate_and_return<<<(outbound_n+255)/256,256,0,stream>>>(outbound,returned,indices,mode?expected:mapped,errors);
            CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
            float ms=0;CHECK(cudaEventElapsedTime(&ms,begin,end));if(r>5)times.push_back(ms*1000);
            int32_t error=0;CHECK(cudaMemcpy(&error,errors,4,cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(result.data(),returned,result.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ids.data(),indices,ids.size()*4,cudaMemcpyDeviceToHost));
            if(error) {std::fprintf(stderr,"FAIL: input mode=%d round=%u\n",mode,r);return 3;}
            for(size_t i=0;i<result.size();++i)if(result[i]!=static_cast<float>(r+i%17))return 4;
            for(size_t i=0;i<ids.size();++i)if(ids[i]!=static_cast<int32_t>(r+i))return 5;
        }
        std::sort(times.begin(),times.end());
        std::printf("PASS expected=%s measured_repeats=%zu warmups=5 kernel_p50_us=%.3f kernel_p99_us=%.3f min=%.3f max=%.3f\n",mode?"device":"host-mapped",times.size(),times[(times.size()-1)/2],times[(times.size()-1)*99/100],times.front(),times.back());
        std::fflush(stdout);
    }
    CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(expected));CHECK(cudaFree(outbound));CHECK(cudaFree(returned));CHECK(cudaFree(indices));CHECK(cudaFree(errors));CHECK(cudaHostUnregister(host));munmap(host,4096);
}
