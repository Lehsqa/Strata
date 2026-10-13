#pragma once
#include "shm_region.hpp"
#include <climits>
#include <string>

namespace xvendor {
struct Arguments {
    std::string role, mode = "futex";
    int fd = -1;
    uint32_t rounds = 10000, timeout_ms = 10000, delay_us = 50;
    size_t registration_mib = 32768, arena_mib = 16384;
    bool region_bytes = false;
    size_t bytes() const {
        return mode == "large-registration" ? checked_mul(registration_mib, size_t{1024 * 1024}) : Layout::standard().bytes;
    }
};
inline uint64_t integer(const std::string& value, uint64_t low, uint64_t high) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer");
    size_t end = 0;
    const uint64_t number = std::stoull(value, &end);
    if (end != value.size() || number < low || number > high) throw std::invalid_argument("integer outside permitted range");
    return number;
}
inline Arguments arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--region-bytes") { result.region_bytes = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value(argv[i]);
        if (key == "--role") result.role = value;
        else if (key == "--mode") result.mode = value;
        else if (key == "--fd") result.fd = static_cast<int>(integer(value, 0, INT_MAX));
        else if (key == "--rounds") result.rounds = static_cast<uint32_t>(integer(value, 1, (1u << 24) - 32));
        else if (key == "--timeout-ms") result.timeout_ms = static_cast<uint32_t>(integer(value, 1, 86400000));
        else if (key == "--delay-us") result.delay_us = static_cast<uint32_t>(integer(value, 0, 1000000));
        else if (key == "--registration-mib") result.registration_mib = integer(value, 1, 262144);
        else if (key == "--arena-mib") result.arena_mib = integer(value, 1, 262144);
        else throw std::invalid_argument("unknown option " + key);
    }
    if (result.mode != "spin" && result.mode != "futex" && result.mode != "doorbell" && result.mode != "large-registration")
        throw std::invalid_argument("unknown mode");
    if (!result.region_bytes && ((result.role != "main" && result.role != "worker") || result.fd < 0))
        throw std::invalid_argument("usage: --role main|worker --fd N [--mode spin|futex|doorbell|large-registration]");
    return result;
}
}
