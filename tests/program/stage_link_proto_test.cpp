#include "strata/program/stage_link_proto.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)
namespace sp = strata::program::stage_proto;

static sp::Identity identity() {
    sp::Identity id;
    id.fingerprint = 1234;
    id.layers = 48; id.embd = 2560; id.hc = 4; id.experts = 512; id.top_k = 10; id.vocab = 248320;
    id.context = 150000; id.split = 12; id.chunk = 1024; id.kv = 2;
    return id;
}
static void layout_tests() {
    auto id = identity();
    sp::Header h;
    std::string err;
    h.identity = id;
    h.parent_pid = 2;
    CHECK(sp::make_layout(id, h.layout, err));
    CHECK(h.layout.decode % 4096 == 0 && h.layout.prompt[0] % 4096 == 0);
    CHECK(h.layout.prompt[0] > h.layout.decode && h.layout.prompt[1] > h.layout.prompt[0]);
    CHECK(h.layout.tokens > h.layout.prompt[1] && h.layout.bytes > h.layout.tokens);
    CHECK(sp::validate_header(h, h.layout.bytes, err));
    CHECK(!sp::validate_header(h, sizeof(sp::Header) - 1, err));
    CHECK(!sp::validate_header(h, h.layout.bytes - 1, err));
    ++h.layout.prompt[1];
    CHECK(!sp::validate_header(h, h.layout.bytes, err));
    CHECK(sp::make_layout(id, h.layout, err));
    ++h.protocol;
    CHECK(!sp::validate_header(h, h.layout.bytes, err));
    h.protocol = sp::version;
    --h.header_bytes;
    CHECK(!sp::validate_header(h, h.layout.bytes, err));
    sp::Layout layout;
    id.embd = INT64_MAX;
    CHECK(!sp::make_layout(id, layout, err));
    id = identity(); id.chunk = INT64_MAX;
    CHECK(!sp::make_layout(id, layout, err));
    id = identity(); id.embd = 1ll << 40;
    CHECK(!sp::make_layout(id, layout, err));
    id = identity(); id.split = id.layers;
    CHECK(!sp::make_layout(id, layout, err));
    id = identity(); id.top_k = id.experts + 1;
    CHECK(!sp::make_layout(id, layout, err));
    id = identity(); id.vocab = INT64_MAX;
    CHECK(!sp::make_layout(id, layout, err));
}
static void session_tests() {
    sp::Header h;
    h.identity = identity();
    sp::Session s;
    std::vector<int64_t> tokens(1024, 19);
    std::string err;
    auto next = [&](sp::Command command) {
        h.request = {};
        h.request.command = command;
        h.command_seq.store(s.sequence + 1);
        h.done_seq.store(s.sequence);
    };
    auto accept = [&] {
        CHECK(s.validate(h, tokens.data(), err));
        s.complete(h.request);
        h.done_seq.store(s.sequence);
    };
    next(sp::Command::run); h.request.count = 1;
    CHECK(!s.validate(h, tokens.data(), err));
    next(sp::Command::hello); accept();
    next(sp::Command::hello);
    CHECK(!s.validate(h, tokens.data(), err));
    next(sp::Command::run); h.request.count = 9;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.count = -1;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.count = 8; h.request.tokens[7] = (int32_t) h.identity.vocab;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.tokens[7] = -1;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.tokens[7] = 198; h.request.pos = INT64_MAX;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.pos = 0; accept();
    CHECK(s.pending == 8 && s.position == 0);
    next(sp::Command::run); h.request.count = 1;
    CHECK(!s.validate(h, tokens.data(), err));
    next(sp::Command::commit); h.request.keep = 9;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.keep = 0;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.keep = 3; accept();
    CHECK(s.pending == 0 && s.position == 3);
    next(sp::Command::commit);
    CHECK(!s.validate(h, tokens.data(), err));
    next(sp::Command::prompt); h.request.count = 1024; h.request.pos = 0;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.pos = 3; h.request.slot = 2;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.slot = 1; h.request.count = 1025;
    CHECK(!s.validate(h, tokens.data(), err));
    h.request.count = 1024; tokens.back() = INT64_MAX;
    CHECK(!s.validate(h, tokens.data(), err));
    tokens.back() = 19;
    CHECK(!s.validate(h, nullptr, err));
    accept();
    CHECK(s.position == 1027);
    next(sp::Command::reset); accept();
    CHECK(s.position == 0 && s.pending == 0);
    next(sp::Command::run); h.request.count = 1; accept();
    next(sp::Command::reset); accept();
    CHECK(s.position == 0 && s.pending == 0);
    next(sp::Command::wait); h.command_seq.store(s.sequence + 2);
    CHECK(!s.validate(h, tokens.data(), err));
    h.command_seq.store(s.sequence + 1); h.done_seq.store(s.sequence - 1);
    CHECK(!s.validate(h, tokens.data(), err));
    next(static_cast<sp::Command>(999));
    CHECK(!s.validate(h, tokens.data(), err));
    s.position = h.identity.context;
    next(sp::Command::run); h.request.count = 1; h.request.pos = s.position;
    CHECK(!s.validate(h, tokens.data(), err));
}
static void reply_tests() {
    sp::Reply reply;
    std::string err;
    CHECK(sp::validate_run_reply(reply, 8, 3, 100, err));
    reply.nonfinite_row = 2; reply.nonfinite_pos = 5; reply.nonfinite_mask = (1u << 2) | (1u << 7);
    reply.picks[2] = -1; reply.picks[7] = -1;
    CHECK(sp::validate_run_reply(reply, 8, 3, 100, err));
    CHECK(!sp::validate_run_reply(reply, 7, 3, 100, err));
    reply.nonfinite_mask |= 1u << 1;
    CHECK(!sp::validate_run_reply(reply, 8, 3, 100, err));
    reply.nonfinite_mask = 1u << 2; reply.nonfinite_pos = 6;
    CHECK(!sp::validate_run_reply(reply, 8, 3, 100, err));
    reply = {}; reply.nonfinite_mask = 1;
    CHECK(!sp::validate_run_reply(reply, 8, 3, 100, err));
    reply = {}; reply.picks[7] = 100;
    CHECK(!sp::validate_run_reply(reply, 8, 3, 100, err));
}
#if defined(__linux__)
static void process_tests() {
    void* shared = mmap(nullptr, sizeof(sp::Header), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    CHECK(shared != MAP_FAILED);
    auto* h = new (shared) sp::Header;
    h->identity = identity();
    std::string err;
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (!pid) {
        sp::Session state;
        while (!h->command_seq.load(std::memory_order_acquire)) std::this_thread::yield();
        std::string child_error;
        if (!state.validate(*h, nullptr, child_error)) _exit(3);
        h->reply.vocab = h->identity.vocab;
        h->done_seq.store(1, std::memory_order_release);
        _exit(0);
    }
    h->command_seq.store(1, std::memory_order_release);
    CHECK(sp::wait_reply(*h, pid, 1, 0, 1000, err));
    CHECK(h->reply.vocab == h->identity.vocab);
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    // A child dying before a command reply must fail promptly rather than waiting for the timeout.
    h->command_seq.store(2); h->done_seq.store(1);
    pid = fork(); CHECK(pid >= 0);
    if (!pid) _exit(17);
    const auto before = std::chrono::steady_clock::now();
    CHECK(!sp::wait_reply(*h, pid, 2, 1, 2000, err));
    CHECK(pid == -1 && err.find("exited") != std::string::npos);
    CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(1));
    // A live child that never replies has a bounded timeout and is reaped by the same cleanup as production.
    pid = fork(); CHECK(pid >= 0);
    if (!pid) for (;;) pause();
    CHECK(!sp::wait_reply(*h, pid, 2, 1, 20, err));
    CHECK(err.find("timed out") != std::string::npos && pid > 0);
    CHECK(sp::kill_worker(pid, 1000) && pid == -1);
    h->done_seq.store(99);
    pid = fork(); CHECK(pid >= 0);
    if (!pid) for (;;) pause();
    CHECK(!sp::wait_reply(*h, pid, 2, 1, 1000, err));
    CHECK(err.find("sequence") != std::string::npos);
    CHECK(sp::kill_worker(pid, 1000));
    CHECK(munmap(shared, sizeof(sp::Header)) == 0);
}
#endif
int main() {
    layout_tests();
    session_tests();
    reply_tests();
#if defined(__linux__)
    process_tests();
#endif
    std::puts("stage_link_proto_test: geometry, malformed commands, sequence/reset lifecycle, crash and timeout passed");
}
