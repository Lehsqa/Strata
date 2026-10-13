// Manual CPU self-check; this does not measure GPU handoff feasibility.
#include "host_pingpong.hpp"
#include <cstdio>
#include <exception>
#include <thread>
#include <sys/prctl.h>
#include <sys/wait.h>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class F> static void rejects(F f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}

static void exchange(bool process, unsigned spin_us, bool use_futex = true) {
    auto region = xvendor::Region::create(xvendor::Layout::standard().bytes);
    xvendor::WaitOptions options;
    options.spin_us = spin_us;
    options.use_futex = use_futex;
    options.timeout_ms = 5000;
    constexpr uint32_t rounds = 10000;
    if (process) {
        const pid_t child = fork();
        check(child >= 0, "fork failed");
        if (child == 0) {
            try {
                auto mapped = xvendor::Region::map(region.fd(), region.size());
                xvendor::Peer peer(getppid());
                options.peer_fd = peer.fd();
                options.peer_pid = peer.pid();
                xvendor::host_pingpong(xvendor::Role::worker, mapped, rounds, options);
                _exit(0);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "worker: %s\n", e.what());
                _exit(1);
            }
        }
        try {
            xvendor::Peer peer(child);
            options.peer_fd = peer.fd();
            options.peer_pid = peer.pid();
            const auto result = xvendor::host_pingpong(xvendor::Role::main, region, rounds, options);
            std::printf("CPU fork %s: %u rounds, p50 %.3f us, p99 %.3f us\n",
                        !use_futex ? "spin only" : (spin_us ? "spin then futex" : "futex"), result.rounds,
                        result.p50_us, result.p99_us);
        } catch (...) {
            kill(child, SIGKILL);
            waitpid(child, nullptr, 0);
            throw;
        }
        int status = 0;
        check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "worker failed");
    } else {
        std::exception_ptr failure;
        std::thread worker([&] {
            try { xvendor::host_pingpong(xvendor::Role::worker, region, rounds, options); }
            catch (...) { failure = std::current_exception(); }
        });
        try { xvendor::host_pingpong(xvendor::Role::main, region, rounds, options); }
        catch (...) { worker.join(); throw; }
        worker.join();
        if (failure) std::rethrow_exception(failure);
    }
}

static void peer_death(bool pid_fallback, bool use_futex) {
    auto region = xvendor::Region::create(xvendor::Layout::standard().bytes);
    const pid_t child = fork();
    check(child >= 0, "peer fork failed");
    if (child == 0) { usleep(20000); _exit(0); }
    xvendor::WaitResult result;
    std::chrono::steady_clock::duration elapsed;
    try {
        xvendor::Peer peer(child);
        xvendor::WaitOptions options;
        options.peer_fd = pid_fallback ? -1 : peer.fd();
        options.peer_pid = peer.pid();
        options.timeout_ms = 2000;
        options.use_futex = use_futex;
        const auto begin = std::chrono::steady_clock::now();
        result = xvendor::wait_sequence(region.header().response, 1, options);
        elapsed = std::chrono::steady_clock::now() - begin;
    } catch (...) {
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        throw;
    }
    if (result != xvendor::WaitResult::peer_exited) kill(child, SIGKILL);
    check(waitpid(child, nullptr, 0) == child, "peer detection reaped child");
    check(result == xvendor::WaitResult::peer_exited, "peer exit not detected");
    check(elapsed < std::chrono::seconds(1), "peer exit detection too slow");
}

static void sibling_zombie() {
    const pid_t victim = fork();
    check(victim >= 0, "sibling victim fork failed");
    if (victim == 0) {
        prctl(PR_SET_NAME, "space ) ( name", 0, 0, 0);
        _exit(0);
    }
    pid_t observer = -1;
    try {
        siginfo_t info{};
        check(waitid(P_PID, static_cast<id_t>(victim), &info, WEXITED | WNOWAIT) == 0,
              "cannot hold sibling zombie");
        observer = fork();
        check(observer >= 0, "sibling observer fork failed");
        if (observer == 0) {
            try {
                auto region = xvendor::Region::create(xvendor::Layout::standard().bytes);
                xvendor::WaitOptions options;
                options.peer_fd = -1;
                options.peer_pid = victim;
                options.timeout_ms = 100;
                _exit(xvendor::wait_sequence(region.header().request, 1, options) ==
                      xvendor::WaitResult::peer_exited ? 0 : 1);
            } catch (...) { _exit(2); }
        }
        int status = 0;
        check(waitpid(observer, &status, 0) == observer, "cannot reap sibling observer");
        observer = -1;
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "sibling zombie was not detected");
    } catch (...) {
        if (observer > 0) { kill(observer, SIGKILL); waitpid(observer, nullptr, 0); }
        kill(victim, SIGKILL);
        waitpid(victim, nullptr, 0);
        throw;
    }
    check(waitpid(victim, nullptr, 0) == victim, "observer reaped sibling zombie");
}

static void publish_then_exit() {
    auto region = xvendor::Region::create(xvendor::Layout::standard().bytes);
    xvendor::WaitOptions options;
    options.spin_us = 0;
    options.timeout_ms = 1000;
    options.use_futex = false;
    for (uint32_t sequence = 1; sequence <= 10000; ++sequence) {
        const pid_t child = fork();
        check(child >= 0, "publishing peer fork failed");
        if (child == 0) {
            region.indices()[0] = static_cast<int32_t>(sequence);
            xvendor::publish(region.header().response, sequence, false);
            _exit(0);
        }
        options.peer_pid = child; // The PID fallback can observe exit during /proc inspection.
        xvendor::WaitResult result;
        try { result = xvendor::wait_sequence(region.header().response, sequence, options); }
        catch (...) { kill(child, SIGKILL); waitpid(child, nullptr, 0); throw; }
        if (result != xvendor::WaitResult::ready) kill(child, SIGKILL);
        int status = 0;
        check(waitpid(child, &status, 0) == child, "cannot reap publishing peer");
        check(result == xvendor::WaitResult::ready, "completed payload lost to peer-exit race");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "publishing peer failed");
        check(region.indices()[0] == static_cast<int32_t>(sequence), "completion lacks acquired payload");
    }
}

int main() {
    try {
        const auto layout = xvendor::Layout::standard();
        const auto page = xvendor::page_size();
        check(layout.outbound_offset % page == 0 && layout.return_offset % page == 0 &&
              layout.indices_offset % page == 0 && layout.bytes % page == 0, "unaligned layout");
        check(layout.outbound_offset >= sizeof(xvendor::SharedHeader), "header overlap");
        check(layout.return_offset >= layout.outbound_offset + xvendor::kOutboundFloats * sizeof(float),
              "outbound overlap");
        check(layout.indices_offset >= layout.return_offset + xvendor::kReturnFloats * sizeof(float),
              "return overlap");
        check(layout.bytes >= layout.indices_offset + xvendor::kRows * sizeof(int32_t), "indices overflow");
        rejects([] { xvendor::checked_add(SIZE_MAX, 1); }, "add overflow accepted");
        rejects([] { xvendor::checked_mul(SIZE_MAX, 2); }, "multiply overflow accepted");
        rejects([] { xvendor::page_align(SIZE_MAX, 4096); }, "alignment overflow accepted");
        rejects([] { xvendor::Region::create(0); }, "empty region accepted");
        rejects([&] { xvendor::Region::create(layout.bytes - 1); }, "unaligned size accepted");
        auto region = xvendor::Region::create(layout.bytes);
        rejects([&] { xvendor::Region::map(region.fd(), layout.bytes + page); }, "short memfd accepted");
        rejects([&] { xvendor::Region::map(-1, layout.bytes); }, "bad fd accepted");
        int copy_fd = -1;
        {
            auto mapped = xvendor::Region::map(region.fd(), layout.bytes);
            copy_fd = mapped.fd();
            check(copy_fd != region.fd(), "mapping did not duplicate fd");
            mapped.outbound()[0] = 123;
            check(region.outbound()[0] == 123, "mappings do not share payload");
            check(reinterpret_cast<uintptr_t>(mapped.data()) % page == 0, "mapping unaligned");
        }
        check(fcntl(copy_fd, F_GETFD) == -1 && errno == EBADF, "mapping leaked fd");
        check(fcntl(region.fd(), F_GETFD) >= 0, "mapping closed borrowed fd");
        {
            auto small = xvendor::Region::create(page);
            rejects([&] { small.outbound(); }, "undersized payload accepted");
            auto large = xvendor::Region::create(layout.bytes + page);
            const int owned = large.fd();
            auto moved = std::move(large);
            check(large.fd() == -1 && moved.fd() == owned, "move did not transfer ownership");
            const int old = small.fd();
            small = std::move(moved);
            check(fcntl(old, F_GETFD) == -1 && errno == EBADF, "move assignment leaked old fd");
            check(small.fd() == owned && small.size() == layout.bytes + page, "flexible region size lost");
            small.outbound()[xvendor::kOutboundFloats - 1] = 42;
        }
        xvendor::WaitOptions options;
        options.spin_us = 0;
        options.timeout_ms = 20;
        auto begin = std::chrono::steady_clock::now();
        check(xvendor::wait_sequence(region.header().request, 1, options) == xvendor::WaitResult::timeout,
              "missing sequence did not time out");
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        check(elapsed >= std::chrono::milliseconds(15) && elapsed < std::chrono::seconds(1),
              "timeout not bounded");
        xvendor::publish(region.header().request, 1);
        check(xvendor::wait_sequence(region.header().request, 1, options) == xvendor::WaitResult::ready,
              "already published value missed");
        options.use_futex = false;
        begin = std::chrono::steady_clock::now();
        check(xvendor::wait_sequence(region.header().response, 1, options) == xvendor::WaitResult::timeout,
              "spin-only wait did not time out");
        check(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1), "spin timeout not bounded");
        rejects([&] { xvendor::host_pingpong(xvendor::Role::main, region, (1u << 24) - 31, options); },
                "inexact float counter accepted");
        peer_death(false, true);
        peer_death(true, true);
        peer_death(true, false);
        sibling_zombie();
        publish_then_exit();
        exchange(true, 50, false);
        exchange(true, 50);
        exchange(true, 0);
        exchange(false, 50);
        std::puts("PASS: shared layout, fd ownership, timeout, peer exit, and fresh payloads");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
