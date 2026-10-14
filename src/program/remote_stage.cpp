#include "remote_stage.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace strata::program {
namespace sp = stage_proto;

sp::Backend stage_backend() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    return sp::Backend::hip;
#else
    return sp::Backend::cuda;
#endif
}

bool stage_model_fingerprint(const std::string& pack, const std::vector<std::string>& shards,
                             uint64_t& fingerprint, std::string& err) {
    uint64_t hash = 1469598103934665603ull;
    auto add = [&](const char* data, size_t n) {
        for (size_t i = 0; i < n; ++i) hash = (hash ^ (unsigned char) data[i]) * 1099511628211ull;
    };
    for (const char* manifest : {"index.txt", "native_experts.txt"}) {
        const std::string path = pack + "/" + manifest;
        std::ifstream file(path, std::ios::binary);
        if (!file) { err = "remote stage: cannot fingerprint " + path; return false; }
        char buffer[8192];
        while (file.read(buffer, sizeof buffer) || file.gcount()) add(buffer, (size_t) file.gcount());
        if (!file.eof()) { err = "remote stage: reading metadata failed: " + path; return false; }
    }
    for (const auto& path : shards) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() < 0) { err = "remote stage: cannot fingerprint " + path; return false; }
        const uint64_t bytes = (uint64_t) file.tellg();
        add(reinterpret_cast<const char*>(&bytes), sizeof bytes);
        file.seekg(0);
        // The GGUF prefix carries model metadata; hashing tensor data would read tens of GiB at startup.
        std::vector<char> prefix((size_t) std::min<uint64_t>(bytes, 1u << 20));
        if (!file.read(prefix.data(), (std::streamsize) prefix.size())) {
            err = "remote stage: short GGUF metadata read: " + path;
            return false;
        }
        add(prefix.data(), prefix.size());
    }
    fingerprint = hash ? hash : 1;
    return true;
}

StageMemory::~StageMemory() {
    if (device_) (void) cudaHostUnregister(static_cast<char*>(mapping_) + layout_.decode);
#if defined(__linux__)
    if (mapping_) munmap(mapping_, (size_t) bytes_);
    if (fd_ >= 0) close(fd_);
#endif
}
std::unique_ptr<StageMemory> StageMemory::create(const sp::Identity& id, std::string& err) {
#if defined(__linux__)
    sp::Layout layout;
    if (!sp::make_layout(id, layout, err)) return nullptr;
    auto memory = std::unique_ptr<StageMemory>(new StageMemory);
    memory->fd_ = (int) syscall(SYS_memfd_create, "strata-stage", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memory->fd_ < 0 || ftruncate(memory->fd_, (off_t) layout.bytes) != 0 ||
        fcntl(memory->fd_, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
        err = std::string("remote stage: creating sealed memfd: ") + std::strerror(errno);
        return nullptr;
    }
    void* mapping = mmap(nullptr, (size_t) layout.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, memory->fd_, 0);
    if (mapping == MAP_FAILED) { err = "remote stage: shared mapping failed"; return nullptr; }
    memory->mapping_ = mapping;
    memory->bytes_ = layout.bytes;
    memory->identity_ = id;
    memory->layout_ = layout;
    auto* h = new (mapping) sp::Header;
    h->identity = id;
    h->layout = layout;
    h->parent_pid = getpid();
    return memory;
#else
    (void) id;
    err = "remote stages require Linux";
    return nullptr;
#endif
}
std::unique_ptr<StageMemory> StageMemory::attach(int fd, std::string& err) {
#if defined(__linux__)
    auto memory = std::unique_ptr<StageMemory>(new StageMemory);
    memory->fd_ = fd;
    struct stat st{};
    const int seals = fcntl(fd, F_GET_SEALS);
    if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(sp::Header) ||
        (uint64_t) st.st_size > (4ull << 30) || seals < 0 ||
        (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) != (F_SEAL_GROW | F_SEAL_SHRINK)) {
        err = "stage worker: fd is not a size-sealed stage memfd";
        return nullptr;
    }
    memory->bytes_ = (uint64_t) st.st_size;
    void* mapping = mmap(nullptr, (size_t) st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) { err = "stage worker: mapping failed"; return nullptr; }
    memory->mapping_ = mapping;
    if (!sp::validate_header(memory->header(), memory->bytes_, err)) return nullptr;
    memory->identity_ = memory->header().identity;
    memory->layout_ = memory->header().layout;
    if (getppid() != memory->header().parent_pid || prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 ||
        getppid() != memory->header().parent_pid) {
        err = "stage worker: parent exited before worker startup";
        return nullptr;
    }
    (void) fcntl(fd, F_SETFD, FD_CLOEXEC);
    return memory;
#else
    (void) fd;
    err = "remote stages require Linux";
    return nullptr;
#endif
}
bool StageMemory::register_gpu(std::string& err) {
    if (device_) return true;
    void* host = static_cast<char*>(mapping_) + layout_.decode;
    const size_t bytes = (size_t) (bytes_ - layout_.decode);
    cudaError_t status = cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (status != cudaSuccess) {
        err = std::string("remote stage: registering shared handoffs: ") + cudaGetErrorString(status);
        return false;
    }
    status = cudaHostGetDevicePointer(&device_, host, 0);
    if (status != cudaSuccess) {
        (void) cudaHostUnregister(host);
        device_ = nullptr;
        err = std::string("remote stage: mapping shared handoffs to device: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}
float* StageMemory::decode_device() const { return static_cast<float*>(device_); }
const float* StageMemory::decode_host() const {
    return reinterpret_cast<const float*>(static_cast<const char*>(mapping_) + layout_.decode);
}
float* StageMemory::prompt(int slot) const {
    return slot < 0 || slot > 1 ? nullptr : reinterpret_cast<float*>(static_cast<char*>(mapping_) + layout_.prompt[slot]);
}
int64_t* StageMemory::prompt_tokens() const {
    return reinterpret_cast<int64_t*>(static_cast<char*>(mapping_) + layout_.tokens);
}
bool StageMemory::unchanged(std::string& err) const {
    if (!sp::same_identity(identity_, header().identity) || !sp::same_layout(layout_, header().layout) ||
        !sp::validate_header(header(), bytes_, err)) {
        if (err.empty()) err = "stage protocol: metadata changed after mapping";
        return false;
    }
    return true;
}

struct RemoteStage::Impl {
    std::unique_ptr<StageMemory> memory;
    std::mutex mutex;
    sp::Session session;
    sp::Reply last;
    sp::Reply last_run;
    std::string failed, deferred;
    bool head_sampling = true;
#if defined(__linux__)
    pid_t pid = -1;
    void kill_worker() {
        if (!sp::kill_worker(pid)) {
            // A driver-stuck child may still hold a GPU operation; leave its pages mapped until process exit.
            (void) memory.release();
            std::fprintf(stderr, "strata remote: killed worker did not exit; shared handoff retained until exit\n");
        }
    }
#endif
    bool exchange(const sp::Request& request, std::string& err) {
        if (!failed.empty()) { err = failed; return false; }
#if defined(__linux__)
        auto fail = [&](const std::string& reason) {
            failed = reason;
            err = reason;
            kill_worker();
            return false;
        };
        auto& h = memory->header();
        if (!memory->unchanged(err)) return fail(err);
        h.request = request;
        // Validate with the next sequence before publication; the worker acquires the actual publication.
        const uint64_t next = session.sequence + 1;
        if (!next || h.done_seq.load(std::memory_order_acquire) != session.sequence)
            return fail("remote stage: sequence exhausted or reply ownership lost");
        h.command_seq.store(next, std::memory_order_release);
        if (!sp::wait_reply(h, pid, next, session.sequence,
                            request.command == sp::Command::hello ? 600000 : 300000, err)) return fail(err);
        last = h.reply;
        if (!memory->unchanged(err)) return fail(err);
        if (last.status) return fail(std::string("remote stage: ") + std::string(last.error, strnlen(last.error, sizeof last.error)));
        session.complete(request);
        return true;
#else
        (void) request;
        err = "remote stages require Linux";
        return false;
#endif
    }
};
RemoteStage::RemoteStage() : impl_(new Impl) {}
RemoteStage::~RemoteStage() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if defined(__linux__)
    if (impl_->pid > 0) {
        std::string err;
        sp::Request request;
        request.command = sp::Command::shutdown;
        (void) impl_->exchange(request, err);
        impl_->kill_worker();
    }
#endif
}
bool RemoteStage::launch(std::unique_ptr<StageMemory> memory, const std::string& executable,
                                             sp::Backend backend, int device,
                                             const std::vector<std::string>& arguments, std::string& err) {
#if defined(__linux__)
    if (!memory || executable.empty() || device < 0) { err = "remote stage: invalid worker launch"; return false; }
    impl_->memory = std::move(memory);
    auto& m = *impl_->memory;
    if (!m.register_gpu(err)) return false;
    std::vector<std::string> args{executable};
    args.insert(args.end(), arguments.begin(), arguments.end());
    // fd 198 belongs only to the worker; the parent's descriptor retains close-on-exec.
    constexpr int worker_fd = 198;
    args.insert(args.end(), {"--stage-worker", "fd:" + std::to_string(worker_fd)});
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    std::vector<std::string> env;
    for (char** p = environ; *p; ++p) {
        const std::string value(*p);
        const std::string key = value.substr(0, value.find('='));
        if (key == "CUDA_VISIBLE_DEVICES" || key == "CUDA_DEVICE_ORDER" || key == "HIP_VISIBLE_DEVICES" ||
            key == "ROCR_VISIBLE_DEVICES" || key == "GPU_DEVICE_ORDINAL" || key == "STRATA_STAGE_LD_LIBRARY_PATH") continue;
        if (key == "LD_LIBRARY_PATH" && std::getenv("STRATA_STAGE_LD_LIBRARY_PATH")) continue;
        env.push_back(value);
    }
    if (backend == sp::Backend::hip) env.push_back("HIP_VISIBLE_DEVICES=" + std::to_string(device));
    else {
        env.push_back("CUDA_DEVICE_ORDER=PCI_BUS_ID");
        env.push_back("CUDA_VISIBLE_DEVICES=" + std::to_string(device));
    }
    if (const char* ld = std::getenv("STRATA_STAGE_LD_LIBRARY_PATH")) env.push_back(std::string("LD_LIBRARY_PATH=") + ld);
    std::vector<char*> envp;
    for (auto& e : env) envp.push_back(e.data());
    envp.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int status = posix_spawn_file_actions_init(&actions);
    if (status) { err = "remote stage: spawn file actions failed"; return false; }
    status = posix_spawn_file_actions_adddup2(&actions, m.fd(), worker_fd);
    if (!status) status = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
    if (!status) status = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (!status) status = posix_spawn(&impl_->pid, executable.c_str(), &actions, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    if (status) { impl_->pid = -1; err = std::string("remote stage: launching worker: ") + std::strerror(status); return false; }
    sp::Request request;
    if (!impl_->exchange(request, err)) return false;
    const auto& reply = impl_->last;
    const auto& id = m.header().identity;
    if (reply.backend != backend || reply.vocab != id.vocab || reply.chunk != id.chunk || reply.cache_slots < 0 ||
        !std::memchr(reply.device, 0, sizeof reply.device) || !std::memchr(reply.arch, 0, sizeof reply.arch)) {
        err = "remote stage: worker backend, vocabulary or prompt capacity differs";
        return false;
    }
    std::fprintf(stderr, "strata remote: READY backend=%s device=%s arch=%s layers=[%lld,%lld) context=%lld "
                         "chunk=%lld cache=%lld free_vram_mib=%llu\n",
                 backend == sp::Backend::hip ? "hip" : "cuda", reply.device, reply.arch,
                 (long long) id.split, (long long) id.layers, (long long) id.context,
                 (long long) reply.chunk, (long long) reply.cache_slots, (unsigned long long) (reply.free_vram >> 20));
    return true;
#else
    (void) memory; (void) executable; (void) backend; (void) device; (void) arguments;
    err = "remote stages require Linux";
    return false;
#endif
}
float* RemoteStage::decode_device() const { return impl_->memory->decode_device(); }
bool RemoteStage::reset(std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    sp::Request r;
    r.command = sp::Command::reset;
    if (!impl_->exchange(r, err)) return false;
    impl_->last_run = {};
    return true;
}
bool RemoteStage::run(int T, const int32_t* tokens, int64_t pos0, core::PoolMultiFn, void*, int32_t* out, std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->deferred.empty()) { err = impl_->deferred; return false; }
    const auto& id = impl_->memory->header().identity;
    if (T < 1 || T > (int) sp::max_window || !tokens || !out || impl_->session.pending ||
        pos0 != impl_->session.position || pos0 < 0 || pos0 > id.context - T) {
        err = "remote stage: invalid RUN size or position";
        return false;
    }
    sp::Request r;
    r.command = sp::Command::run;
    r.count = T;
    r.pos = pos0;
    r.head_sampling = impl_->head_sampling;
    std::copy(tokens, tokens + T, r.tokens);
    for (int t = 0; t < T; ++t) if (tokens[t] < 0 || tokens[t] >= id.vocab) {
        err = "remote stage: RUN token outside vocabulary";
        return false;
    }
    if (!impl_->exchange(r, err)) return false;
    const auto& reply = impl_->last;
    if (!sp::validate_run_reply(reply, T, pos0, id.vocab, err)) return false;
    impl_->last_run = reply;
    std::copy(reply.picks, reply.picks + T, out);
    return true;
}
bool RemoteStage::commit(int keep, std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->session.pending || keep < 1 || keep > impl_->session.pending) {
        err = "remote stage: COMMIT outside the pending window";
        return false;
    }
    sp::Request r;
    r.command = sp::Command::commit;
    r.keep = keep;
    return impl_->exchange(r, err);
}
bool RemoteStage::wait_commit(std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    sp::Request r;
    r.command = sp::Command::wait;
    return impl_->exchange(r, err);
}
bool RemoteStage::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!bias.empty()) { err = "remote stage v1 does not support logit bias"; return false; }
    return true;
}
void RemoteStage::set_sampling(const kernels::SamplerParams& p) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!std::isfinite(p.temperature) || !std::isfinite(p.top_p) || !std::isfinite(p.min_p) ||
        (!p.greedy && p.temperature > 0) || p.logit_bias || p.penalty_last_n ||
        p.penalty_repeat != 1 || p.penalty_freq != 0 || p.penalty_present != 0 || p.gumbel)
        impl_->deferred = "remote stage v1 requires greedy sampling without penalties, bias or speculative probabilities";
}
void RemoteStage::set_history(const int32_t* history, int len) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (history || len) impl_->deferred = "remote stage v1 does not support penalty history";
}
void RemoteStage::set_spec_q(const int32_t* q, int n_q) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (q || n_q) impl_->deferred = "remote stage v1 does not support speculative probability rows";
}
void RemoteStage::set_head_sampling(bool on) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->head_sampling = on;
}
int64_t RemoteStage::vocab() const { return impl_->memory->header().identity.vocab; }
int RemoteStage::nonfinite_row() const { return impl_->last_run.nonfinite_row; }
int64_t RemoteStage::nonfinite_pos() const { return impl_->last_run.nonfinite_pos; }
bool RemoteStage::nonfinite_in(int t) const {
    return t >= 0 && t < (int) sp::max_window && (impl_->last_run.nonfinite_mask & (1u << t));
}
float* RemoteStage::chunk_slot(int slot) { return impl_->memory->prompt(slot); }
bool RemoteStage::run_chunk(const int64_t* tokens, int64_t n, int64_t pos0, const float* rows,
                            bool single_chunk, std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto& id = impl_->memory->header().identity;
    const int slot = rows == chunk_slot(0) ? 0 : rows == chunk_slot(1) ? 1 : -1;
    if (!tokens || slot < 0 || n < 1 || n > id.chunk || impl_->session.pending || pos0 != impl_->session.position ||
        pos0 < 0 || pos0 > id.context - n) {
        err = "remote stage: invalid prompt chunk, slot or position";
        return false;
    }
    std::copy(tokens, tokens + n, impl_->memory->prompt_tokens());
    sp::Request r;
    r.command = sp::Command::prompt;
    r.count = (int32_t) n;
    r.slot = slot;
    r.pos = pos0;
    r.single_chunk = single_chunk;
    if (!impl_->exchange(r, err)) return false;
    if (on_prompt_done) on_prompt_done(pos0 + n);
    return true;
}
bool RemoteStage::drain(std::string& err) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    sp::Request r;
    r.command = sp::Command::drain;
    return impl_->exchange(r, err);
}

int stage_worker_loop(StageMemory& memory, core::Verifier& verifier, prefill::Prefill& prefill,
                      core::PoolMultiFn pool, void* user, int64_t chunk, int64_t cache_slots,
                      const std::function<bool(std::string&)>& reset,
                      const std::function<bool(std::string&)>& pool_ok) {
#if defined(__linux__)
    auto& h = memory.header();
    sp::Session session;
    for (;;) {
        while (h.command_seq.load(std::memory_order_acquire) == session.sequence)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        sp::Reply reply;
        std::string err;
        bool ok = memory.unchanged(err) && session.validate(h, memory.prompt_tokens(), err);
        const auto request = h.request;
        if (ok) {
            try {
                switch (request.command) {
                case sp::Command::hello: {
                    cudaDeviceProp props{};
                    int dev = 0;
                    size_t free_bytes = 0, total_bytes = 0;
                    ok = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&props, dev) == cudaSuccess &&
                         cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
                    if (!ok) { err = "worker GPU properties unavailable"; break; }
                    reply.backend = stage_backend();
                    reply.vocab = verifier.vocab();
                    reply.chunk = chunk;
                    reply.cache_slots = cache_slots;
                    reply.free_vram = free_bytes;
                    std::snprintf(reply.device, sizeof reply.device, "%s", props.name);
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
                    std::snprintf(reply.arch, sizeof reply.arch, "%s", props.gcnArchName);
#else
                    std::snprintf(reply.arch, sizeof reply.arch, "sm_%d%d", props.major, props.minor);
#endif
                    if (reply.backend != h.identity.backend || reply.vocab != h.identity.vocab || chunk < h.identity.chunk) {
                        ok = false;
                        err = "worker geometry or backend differs from HELLO";
                    }
                    break;
                }
                case sp::Command::run: {
                    const uint64_t count = (uint64_t) request.count *
                        ((uint64_t) h.identity.hc * (uint64_t) h.identity.embd + (uint64_t) h.identity.embd + (uint64_t) h.identity.hc);
                    for (uint64_t i = 0; i < count; ++i) if (!std::isfinite(memory.decode_host()[i])) {
                        ok = false;
                        err = "worker received nonfinite decode handoff";
                        break;
                    }
                    if (!ok) break;
                    verifier.set_head_sampling(request.head_sampling != 0);
                    ok = verifier.run(request.count, request.tokens, request.pos, pool, user, reply.picks, err) && pool_ok(err);
                    reply.nonfinite_row = verifier.nonfinite_row();
                    reply.nonfinite_pos = verifier.nonfinite_pos();
                    for (int t = 0; t < request.count; ++t)
                        if (verifier.nonfinite_in(t)) reply.nonfinite_mask |= 1u << t;
                    break;
                }
                case sp::Command::commit:
                    ok = verifier.commit(request.keep, err) && verifier.wait_commit(err);
                    break;
                case sp::Command::wait:
                    ok = verifier.wait_commit(err);
                    break;
                case sp::Command::prompt: {
                    const float* rows = memory.prompt(request.slot);
                    const uint64_t count = (uint64_t) request.count * (uint64_t) h.identity.hc * (uint64_t) h.identity.embd;
                    for (uint64_t i = 0; i < count; ++i) if (!std::isfinite(rows[i])) {
                        err = "worker received nonfinite prompt handoff";
                        ok = false;
                        break;
                    }
                    if (ok) ok = prefill.run_stage_chunk(memory.prompt_tokens(), request.count, request.pos,
                                                         rows, request.single_chunk != 0, err);
                    break;
                }
                case sp::Command::reset:
                    ok = verifier.wait_commit(err) && prefill.drain_pipeline(err) && reset(err);
                    break;
                case sp::Command::drain: case sp::Command::shutdown:
                    ok = verifier.wait_commit(err) && prefill.drain_pipeline(err);
                    break;
                }
            } catch (const std::exception& exception) { ok = false; err = exception.what(); }
        }
        if (!ok) sp::fail(reply, err.empty() ? "worker command failed" : err);
        h.reply = reply;
        // A malformed command still gets an owned failure reply; the worker exits afterward.
        h.done_seq.store(h.command_seq.load(std::memory_order_acquire), std::memory_order_release);
        if (!ok) return 1;
        session.complete(request);
        if (request.command == sp::Command::shutdown) return 0;
    }
#else
    (void) memory; (void) verifier; (void) prefill; (void) pool; (void) user; (void) chunk; (void) cache_slots; (void) reset; (void) pool_ok;
    return 2;
#endif
}

} // namespace strata::program
