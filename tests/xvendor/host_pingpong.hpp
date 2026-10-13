#pragma once
#include "shm_region.hpp"
#include <vector>

namespace xvendor {

enum class Role { main, worker };
struct Latencies {
    uint32_t rounds = 0;
    double p50_us = 0;
    double p99_us = 0;
};

inline void require_ready(WaitResult result) {
    if (result == WaitResult::timeout) throw std::runtime_error("handoff timed out");
    if (result == WaitResult::peer_exited) throw std::runtime_error("handoff peer exited");
}

inline Latencies host_pingpong(Role role, Region& region, uint32_t rounds, const WaitOptions& options = {}) {
    if (!rounds || rounds > (1u << 24) - 32)
        throw std::invalid_argument("host rounds must fit exact float payload counter");
    auto& header = region.header();
    float* outbound = region.outbound();
    float* returned = region.returned();
    int32_t* indices = region.indices();
    std::vector<double> timings;
    if (role == Role::main) timings.reserve(rounds);
    for (uint32_t i = 0; i < rounds; ++i) {
        const uint32_t sequence = i + 1;
        if (role == Role::main) {
            // CPU baseline includes filling and checking the entire handoff payload.
            const auto begin = std::chrono::steady_clock::now();
            for (size_t j = 0; j < kOutboundFloats; ++j) outbound[j] = static_cast<float>(sequence + j % 13);
            publish(header.request, sequence, options.use_futex);
            require_ready(wait_sequence(header.response, sequence, options));
            for (size_t j = 0; j < kReturnFloats; ++j)
                if (returned[j] != static_cast<float>(sequence + j % 17))
                    throw std::runtime_error("stale CPU return payload");
            for (size_t j = 0; j < kRows; ++j)
                if (indices[j] != static_cast<int32_t>(sequence + j))
                    throw std::runtime_error("stale CPU indices");
            timings.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - begin).count());
        } else {
            require_ready(wait_sequence(header.request, sequence, options));
            for (size_t j = 0; j < kOutboundFloats; ++j)
                if (outbound[j] != static_cast<float>(sequence + j % 13))
                    throw std::runtime_error("stale CPU outbound payload");
            for (size_t j = 0; j < kReturnFloats; ++j) returned[j] = static_cast<float>(sequence + j % 17);
            for (size_t j = 0; j < kRows; ++j) indices[j] = static_cast<int32_t>(sequence + j);
            publish(header.response, sequence, options.use_futex);
        }
    }
    Latencies result;
    result.rounds = rounds;
    if (!timings.empty()) {
        std::sort(timings.begin(), timings.end());
        result.p50_us = timings[(timings.size() - 1) * 50 / 100];
        result.p99_us = timings[(timings.size() - 1) * 99 / 100];
    }
    return result;
}

} // namespace xvendor
