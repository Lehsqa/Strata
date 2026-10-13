#pragma once
#include <stddef.h>

// No engine types or allocator ownership cross the shared-library boundary.
struct XProbeAPI {
    unsigned abi_version;
    int (*init)();
    int (*register_host)(void*, size_t);
    int (*check)();
    int (*unregister_host)();
    int (*shutdown)();
    const char* (*error)();
};
using XProbeEntry = const XProbeAPI* (*)();
