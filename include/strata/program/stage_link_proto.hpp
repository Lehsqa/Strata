#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#if defined(__linux__)
#include <cerrno>
#include <chrono>
#include <csignal>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace strata::program::stage_proto {

constexpr uint64_t magic = 0x5354524154415331ull;
constexpr uint32_t version = 1;
constexpr uint64_t max_window = 8;
enum class Command : uint32_t { hello = 1, run, commit, wait, prompt, drain, reset, shutdown };
enum class Backend : uint32_t { cuda = 1, hip = 2 };

struct Identity {
    uint64_t fingerprint = 0;
    int64_t layers = 0, embd = 0, hc = 0, experts = 0, top_k = 0, vocab = 0;
    int64_t context = 0, split = 0, chunk = 0;
    uint32_t kv = 0;
    Backend backend = Backend::cuda;
};
inline bool same_identity(const Identity& a, const Identity& b) {
    return a.fingerprint == b.fingerprint && a.layers == b.layers && a.embd == b.embd &&
           a.hc == b.hc && a.experts == b.experts && a.top_k == b.top_k && a.vocab == b.vocab &&
           a.context == b.context && a.split == b.split && a.chunk == b.chunk && a.kv == b.kv &&
           a.backend == b.backend;
}
struct Layout {
    uint64_t bytes = 0, decode = 0, prompt[2] = {}, tokens = 0;
    uint64_t decode_floats = 0, prompt_floats = 0;
};
struct Request {
    Command command = Command::hello;
    int32_t count = 0, keep = 0, slot = 0, single_chunk = 0, head_sampling = 1;
    int64_t pos = 0;
    int32_t tokens[max_window] = {};
};
struct Reply {
    int32_t status = 0, picks[max_window] = {}, nonfinite_row = -1;
    uint32_t nonfinite_mask = 0;
    int64_t nonfinite_pos = -1, vocab = 0, chunk = 0, cache_slots = 0;
    uint64_t free_vram = 0;
    Backend backend = Backend::cuda;
    char device[128] = {}, arch[48] = {}, error[512] = {};
};
struct alignas(64) Header {
    uint64_t signature = magic;
    uint32_t protocol = version, header_bytes = sizeof(Header);
    Identity identity;
    Layout layout;
    int64_t parent_pid = 0;
    alignas(64) std::atomic<uint64_t> command_seq{0};
    alignas(64) std::atomic<uint64_t> done_seq{0};
    Request request;
    Reply reply;
};
static_assert(std::atomic<uint64_t>::is_always_lock_free, "shared stage sequence must be lock-free");

inline bool product(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}
inline bool make_layout(const Identity& id, Layout& out, std::string& err) {
    if (!id.fingerprint || id.layers < 3 || id.embd <= 0 || id.hc <= 0 || id.experts <= 0 ||
        id.top_k <= 0 || id.top_k > id.experts || id.vocab <= 0 || id.vocab > INT32_MAX ||
        id.context <= 0 || id.split < 2 || id.split >= id.layers || id.chunk <= 0 ||
        id.chunk > id.context || id.chunk > INT32_MAX || id.kv > 2 ||
        (id.backend != Backend::cuda && id.backend != Backend::hip)) {
        err = "stage protocol: invalid model geometry or capacities";
        return false;
    }
    uint64_t width = 0, residual = 0;
    if (!product((uint64_t) id.embd, (uint64_t) id.hc, residual) ||
        residual > UINT64_MAX - (uint64_t) id.embd - (uint64_t) id.hc ||
        !product(max_window, residual + (uint64_t) id.embd + (uint64_t) id.hc, width)) {
        err = "stage protocol: geometry overflow";
        return false;
    }
    out = {};
    out.decode_floats = width;
    if (!product((uint64_t) id.chunk, residual, out.prompt_floats)) {
        err = "stage protocol: prompt capacity overflow";
        return false;
    }
    uint64_t next = (sizeof(Header) + 4095u) & ~uint64_t(4095u);
    auto append = [&](uint64_t count, uint64_t item, uint64_t& offset) {
        uint64_t size = 0;
        if (!product(count, item, size) || size > UINT64_MAX - next - 4095u) return false;
        offset = next;
        next = (next + size + 4095u) & ~uint64_t(4095u);
        return true;
    };
    if (!append(width, sizeof(float), out.decode) ||
        !append(out.prompt_floats, sizeof(float), out.prompt[0]) ||
        !append(out.prompt_floats, sizeof(float), out.prompt[1]) ||
        !append((uint64_t) id.chunk, sizeof(int64_t), out.tokens) || next > (4ull << 30) || next > SIZE_MAX) {
        err = "stage protocol: shared region exceeds 4 GiB";
        return false;
    }
    out.bytes = next;
    return true;
}
inline bool same_layout(const Layout& a, const Layout& b) {
    return a.bytes == b.bytes && a.decode == b.decode && a.prompt[0] == b.prompt[0] &&
           a.prompt[1] == b.prompt[1] && a.tokens == b.tokens &&
           a.decode_floats == b.decode_floats && a.prompt_floats == b.prompt_floats;
}
inline bool validate_header(const Header& h, uint64_t bytes, std::string& err) {
    Layout expected;
    if (h.signature != magic || h.protocol != version || h.header_bytes != sizeof(Header) ||
        h.parent_pid <= 1 || !make_layout(h.identity, expected, err) ||
        !same_layout(h.layout, expected) || h.layout.bytes != bytes) {
        if (err.empty()) err = "stage protocol: incompatible header or mapping size";
        return false;
    }
    return true;
}

// Each RUN is speculative until COMMIT; prompt chunks advance the session immediately.
struct Session {
    int64_t position = 0;
    int32_t pending = 0;
    bool hello = false;
    uint64_t sequence = 0;
    bool validate(const Header& h, const int64_t* prompt_tokens, std::string& err) const {
        const auto& r = h.request;
        const auto& id = h.identity;
        if (r.head_sampling != 0 && r.head_sampling != 1) {
            err = "stage protocol: invalid head sampling flag";
            return false;
        }
        if (sequence == UINT64_MAX || h.command_seq.load(std::memory_order_acquire) != sequence + 1 ||
            h.done_seq.load(std::memory_order_acquire) != sequence) {
            err = "stage protocol: unexpected command sequence";
            return false;
        }
        if ((!hello && r.command != Command::hello) || (hello && r.command == Command::hello)) {
            err = "stage protocol: HELLO required exactly once";
            return false;
        }
        switch (r.command) {
        case Command::hello: case Command::wait: case Command::drain: case Command::reset: case Command::shutdown:
            if (r.count || r.keep || r.pos || r.slot || r.single_chunk) break;
            return true;
        case Command::run:
            if (pending || r.count < 1 || r.count > (int32_t) max_window || r.keep || r.slot || r.single_chunk ||
                r.pos != position || r.pos < 0 || r.pos > id.context - r.count) break;
            for (int t = 0; t < r.count; ++t)
                if (r.tokens[t] < 0 || r.tokens[t] >= id.vocab) {
                    err = "stage protocol: RUN token outside vocabulary";
                    return false;
                }
            return true;
        case Command::commit:
            if (!pending || r.keep < 1 || r.keep > pending || r.count || r.pos || r.slot || r.single_chunk) break;
            return true;
        case Command::prompt:
            if (pending || r.count < 1 || r.count > id.chunk || r.keep || r.slot < 0 || r.slot > 1 ||
                (r.single_chunk != 0 && r.single_chunk != 1) || r.pos != position ||
                r.pos < 0 || r.pos > id.context - r.count || !prompt_tokens) break;
            for (int t = 0; t < r.count; ++t)
                if (prompt_tokens[t] < 0 || prompt_tokens[t] >= id.vocab) {
                    err = "stage protocol: PROMPT token outside vocabulary";
                    return false;
                }
            return true;
        default: break;
        }
        err = "stage protocol: invalid command fields or session position";
        return false;
    }
    void complete(const Request& r) {
        switch (r.command) {
        case Command::hello: hello = true; break;
        case Command::run: pending = r.count; break;
        case Command::commit: position += r.keep; pending = 0; break;
        case Command::prompt: position += r.count; break;
        case Command::reset: position = 0; pending = 0; break;
        default: break;
        }
        ++sequence;
    }
};
inline void fail(Reply& reply, const std::string& error) {
    reply.status = 1;
    const size_t n = error.size() < sizeof reply.error - 1 ? error.size() : sizeof reply.error - 1;
    std::memcpy(reply.error, error.data(), n);
    reply.error[n] = 0;
}
inline bool validate_run_reply(const Reply& reply, int T, int64_t pos, int64_t vocab, std::string& err) {
    if (T < 1 || T > (int) max_window || pos < 0 || pos > INT64_MAX - T || vocab < 1 ||
        reply.nonfinite_row < -1 || reply.nonfinite_row >= T ||
        (reply.nonfinite_mask >> T) ||
        ((reply.nonfinite_row < 0) != (reply.nonfinite_mask == 0)) ||
        (reply.nonfinite_row >= 0 &&
         (reply.nonfinite_pos != pos + reply.nonfinite_row ||
          !(reply.nonfinite_mask & (1u << reply.nonfinite_row)) ||
          (reply.nonfinite_mask & ((1u << reply.nonfinite_row) - 1u)))) ||
        (reply.nonfinite_row < 0 && reply.nonfinite_pos != -1)) {
        err = "remote stage: invalid nonfinite diagnostics";
        return false;
    }
    for (int t = 0; t < T; ++t) if (!(reply.nonfinite_mask & (1u << t)) &&
        (reply.picks[t] < 0 || reply.picks[t] >= vocab)) {
        err = "remote stage: worker pick outside vocabulary";
        return false;
    }
    return true;
}

#if defined(__linux__)
inline bool wait_reply(Header& h, pid_t& pid, uint64_t sequence, uint64_t previous,
                       int timeout_ms, std::string& err) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const uint64_t done = h.done_seq.load(std::memory_order_acquire);
        if (done == sequence) return true;
        if (done != previous) { err = "remote stage: invalid reply sequence"; return false; }
        int status = 0;
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno != EINTR)) {
            pid = -1;
            err = "remote stage: worker exited before replying";
            return false;
        }
        if (std::chrono::steady_clock::now() >= until) { err = "remote stage: worker command timed out"; return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
inline bool kill_worker(pid_t& pid, int timeout_ms = 10000) {
    if (pid <= 0) return true;
    (void) kill(pid, SIGKILL);
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        int status = 0;
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) { pid = -1; return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (std::chrono::steady_clock::now() < until);
    return false;
}
#endif

} // namespace strata::program::stage_proto
