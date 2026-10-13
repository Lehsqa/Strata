#pragma once
// Linux-only feasibility transport. No CUDA/HIP headers or engine dependencies.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <fcntl.h>
#include <linux/futex.h>
#include <linux/memfd.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace xvendor {

constexpr size_t kRows = 8;
constexpr size_t kOutboundWidth = 12804;
constexpr size_t kReturnWidth = 10240;
constexpr size_t kOutboundFloats = kRows * kOutboundWidth;
constexpr size_t kReturnFloats = kRows * kReturnWidth;

inline std::runtime_error system_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": errno " + std::to_string(errno));
}

inline size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("shared region size overflow");
    return a + b;
}

inline size_t checked_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a) throw std::overflow_error("shared region size overflow");
    return a * b;
}

inline size_t page_size() {
    const long result = sysconf(_SC_PAGESIZE);
    if (result <= 0) throw std::runtime_error("cannot read page size");
    return static_cast<size_t>(result);
}

inline size_t page_align(size_t bytes, size_t page) {
    if (!page) throw std::invalid_argument("zero page size");
    const size_t remainder = bytes % page;
    return remainder ? checked_add(bytes, page - remainder) : bytes;
}

// Raw 32-bit words also let the GPU benchmark pass .value to doorbell kernels.
struct alignas(64) Sequence {
    uint32_t value;
    uint32_t load() const { return __atomic_load_n(&value, __ATOMIC_ACQUIRE); }
};
static_assert(sizeof(uint32_t) == 4 && __atomic_always_lock_free(4, nullptr),
              "shared sequences require lock-free 32-bit atomics");

struct SharedHeader {
    Sequence request;
    Sequence response;
    Sequence gpu_request;
    Sequence gpu_response;
    alignas(64) uint32_t gpu_expected;
};

struct Layout {
    size_t outbound_offset, return_offset, indices_offset, bytes;
    static Layout standard() {
        const size_t page = page_size();
        Layout result{};
        result.outbound_offset = page_align(sizeof(SharedHeader), page);
        result.return_offset = page_align(checked_add(result.outbound_offset,
            checked_mul(kOutboundFloats, sizeof(float))), page);
        result.indices_offset = page_align(checked_add(result.return_offset,
            checked_mul(kReturnFloats, sizeof(float))), page);
        result.bytes = page_align(checked_add(result.indices_offset,
            checked_mul(kRows, sizeof(int32_t))), page);
        return result;
    }
};

class Region {
    int fd_ = -1;
    void* data_ = MAP_FAILED;
    size_t bytes_ = 0;

    static void validate_size(size_t bytes) {
        if (!bytes || bytes % page_size() ||
            bytes > static_cast<size_t>(std::numeric_limits<off_t>::max()))
            throw std::invalid_argument("region size must be nonzero, page-aligned and fit off_t");
    }
    static Region map_owned(int fd, size_t bytes) {
        Region result;
        result.fd_ = fd;
        struct stat info{};
        if (fstat(fd, &info) != 0) throw system_error("fstat memfd");
        if (info.st_size < 0 || static_cast<uintmax_t>(info.st_size) < bytes)
            throw std::invalid_argument("memfd is shorter than mapping");
        result.data_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (result.data_ == MAP_FAILED) throw system_error("mmap memfd");
        result.bytes_ = bytes;
        return result;
    }
    void require_payload() const {
        if (bytes_ < Layout::standard().bytes) throw std::invalid_argument("region has no full payload layout");
    }
public:
    Region() = default;
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;
    Region(Region&& other) noexcept : fd_(other.fd_), data_(other.data_), bytes_(other.bytes_) {
        other.fd_ = -1; other.data_ = MAP_FAILED; other.bytes_ = 0;
    }
    Region& operator=(Region&& other) noexcept {
        if (this != &other) {
            Region old(std::move(*this));
            fd_ = other.fd_; data_ = other.data_; bytes_ = other.bytes_;
            other.fd_ = -1; other.data_ = MAP_FAILED; other.bytes_ = 0;
        }
        return *this;
    }
    ~Region() {
        if (data_ != MAP_FAILED) munmap(data_, bytes_);
        if (fd_ >= 0) close(fd_);
    }
    static Region create(size_t bytes) {
        validate_size(bytes);
        const int fd = static_cast<int>(syscall(SYS_memfd_create, "strata-xvendor", MFD_CLOEXEC));
        if (fd < 0) throw system_error("memfd_create");
        if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            const auto error = system_error("ftruncate memfd");
            close(fd);
            throw error;
        }
        return map_owned(fd, bytes); // Sparse: do not touch a registration-sized mapping.
    }
    static Region map(int fd, size_t bytes) {
        validate_size(bytes);
        const int owned = fcntl(fd, F_DUPFD_CLOEXEC, 0);
        if (owned < 0) throw system_error("duplicate memfd");
        return map_owned(owned, bytes);
    }
    int fd() const { return fd_; }
    void* data() const { return data_; }
    size_t size() const { return bytes_; }
    SharedHeader& header() const {
        if (bytes_ < sizeof(SharedHeader)) throw std::invalid_argument("region has no header");
        return *static_cast<SharedHeader*>(data_);
    }
    float* outbound() const {
        require_payload();
        return reinterpret_cast<float*>(static_cast<char*>(data_) + Layout::standard().outbound_offset);
    }
    float* returned() const {
        require_payload();
        return reinterpret_cast<float*>(static_cast<char*>(data_) + Layout::standard().return_offset);
    }
    int32_t* indices() const {
        require_payload();
        return reinterpret_cast<int32_t*>(static_cast<char*>(data_) + Layout::standard().indices_offset);
    }
};

class Peer {
    pid_t pid_;
    int fd_ = -1;
public:
    explicit Peer(pid_t pid) : pid_(pid) {
        if (pid <= 0) throw std::invalid_argument("peer PID must be positive");
#ifdef SYS_pidfd_open
        fd_ = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
        if (fd_ < 0 && errno != ENOSYS && errno != EINVAL && errno != EPERM &&
            errno != EACCES && errno != ESRCH) throw system_error("pidfd_open");
#endif
    }
    Peer(const Peer&) = delete;
    Peer& operator=(const Peer&) = delete;
    ~Peer() { if (fd_ >= 0) close(fd_); }
    int fd() const { return fd_; }
    pid_t pid() const { return pid_; }
};

enum class WaitResult { ready, timeout, peer_exited };
struct WaitOptions {
    unsigned spin_us = 50;
    unsigned timeout_ms = 1000;
    bool use_futex = true;
    int peer_fd = -1;
    pid_t peer_pid = -1;
};

inline bool peer_exited(const WaitOptions& options) {
    if (options.peer_fd >= 0) {
        pollfd descriptor{options.peer_fd, POLLIN, 0};
        const int result = poll(&descriptor, 1, 0);
        if (result < 0 && errno == EINTR) return false;
        if (result < 0) throw system_error("poll peer");
        if (descriptor.revents & (POLLERR | POLLNVAL)) throw std::runtime_error("invalid peer pidfd");
        return descriptor.revents & (POLLIN | POLLHUP);
    }
    if (options.peer_pid <= 0) return false;
    siginfo_t info{};
    if (waitid(P_PID, static_cast<id_t>(options.peer_pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0 &&
        info.si_pid == options.peer_pid) return true;
    // shortcut: PID fallback can race PID reuse; use pidfds when the kernel allows them.
    if (kill(options.peer_pid, 0) != 0 && errno != EPERM) {
        if (errno == ESRCH) return true;
        throw system_error("check peer PID");
    }
    // Nonchildren can remain zombies while their parent holds the exit status.
    std::ifstream stat("/proc/" + std::to_string(options.peer_pid) + "/stat");
    const std::string text((std::istreambuf_iterator<char>(stat)), std::istreambuf_iterator<char>());
    const size_t end = text.rfind(')'); // comm may contain spaces, newlines and parentheses.
    if (end != std::string::npos && text.size() - end > 2) {
        const char state = text[end + 2];
        return state == 'Z' || state == 'X' || state == 'x';
    }
    return false;
}

inline void publish(Sequence& sequence, uint32_t value, bool wake = true) {
    __atomic_store_n(&sequence.value, value, __ATOMIC_RELEASE);
    if (wake && syscall(SYS_futex, &sequence.value, FUTEX_WAKE, 1, nullptr, nullptr, 0) < 0)
        throw system_error("futex wake");
}

inline WaitResult wait_sequence(Sequence& sequence, uint32_t expected, const WaitOptions& options = {}) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    const auto deadline = begin + std::chrono::milliseconds(options.timeout_ms);
    const auto spin_until = std::min(deadline, begin + std::chrono::microseconds(options.spin_us));
    auto check_peer_at = begin;
    for (;;) {
        const uint32_t observed = sequence.load();
        if (observed == expected) return WaitResult::ready;
        const auto now = Clock::now();
        if (now >= check_peer_at) {
            if (peer_exited(options))
                return sequence.load() == expected ? WaitResult::ready : WaitResult::peer_exited;
            check_peer_at = now + std::chrono::milliseconds(10);
        }
        if (now >= deadline)
            return sequence.load() == expected ? WaitResult::ready : WaitResult::timeout;
        if (!options.use_futex || now < spin_until) continue;
        const auto sleep = std::min(deadline - now,
            std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(10)));
        const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(sleep).count();
        timespec timeout{static_cast<time_t>(nanos / 1000000000), static_cast<long>(nanos % 1000000000)};
        const long result = syscall(SYS_futex, &sequence.value, FUTEX_WAIT, observed, &timeout, nullptr, 0);
        if (result < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT)
            throw system_error("futex wait");
    }
}

} // namespace xvendor
