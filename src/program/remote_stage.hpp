#pragma once
#include "strata/core/stage_link.hpp"
#include "strata/prefill/prefill_link.hpp"
#include "strata/program/stage_link_proto.hpp"
#include <functional>
#include <memory>

namespace strata::core { class Verifier; }
namespace strata::prefill { class Prefill; }
namespace strata::program {

// Metadata only: the model's pack manifests and GGUF headers, never its expert arena.
bool stage_model_fingerprint(const std::string& pack, const std::vector<std::string>& shards,
                             uint64_t& fingerprint, std::string& err);
stage_proto::Backend stage_backend();

class StageMemory {
public:
    ~StageMemory();
    StageMemory(const StageMemory&) = delete;
    StageMemory& operator=(const StageMemory&) = delete;
    static std::unique_ptr<StageMemory> create(const stage_proto::Identity& id, std::string& err);
    static std::unique_ptr<StageMemory> attach(int fd, std::string& err);
    bool register_gpu(std::string& err);
    stage_proto::Header& header() const { return *static_cast<stage_proto::Header*>(mapping_); }
    float* decode_device() const;
    const float* decode_host() const;
    float* prompt(int slot) const;
    int64_t* prompt_tokens() const;
    int fd() const { return fd_; }
    bool unchanged(std::string& err) const;
private:
    StageMemory() = default;
    int fd_ = -1;
    void* mapping_ = nullptr;
    void* device_ = nullptr;
    uint64_t bytes_ = 0;
    stage_proto::Identity identity_;
    stage_proto::Layout layout_;
};

class RemoteStage final : public core::StageLink, public prefill::PrefillLink {
public:
    RemoteStage();
    ~RemoteStage() override;
    bool launch(std::unique_ptr<StageMemory> memory,
                                            const std::string& executable, stage_proto::Backend backend,
                                            int device, const std::vector<std::string>& arguments,
                                            std::string& err);
    float* decode_device() const;
    bool reset(std::string& err);
    bool run(int T, const int32_t* tokens, int64_t pos0, core::PoolMultiFn, void*, int32_t* out,
             std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    bool wait_commit(std::string& err) override;
    bool set_logit_bias(const std::vector<float>& bias, std::string& err) override;
    void set_sampling(const kernels::SamplerParams& sp) override;
    void set_history(const int32_t* history, int len) override;
    void set_spec_q(const int32_t* q, int n_q) override;
    void set_head_sampling(bool on) override;
    bool copy_logits(int, float*) const override { return false; }
    int64_t vocab() const override;
    int nonfinite_row() const override;
    int64_t nonfinite_pos() const override;
    bool nonfinite_in(int t) const override;
    const float* final_R_all() const override { return nullptr; }
    float* chunk_slot(int slot) override;
    bool run_chunk(const int64_t* tokens, int64_t n, int64_t pos0, const float* rows,
                   bool single_chunk, std::string& err) override;
    bool drain(std::string& err) override;
    std::function<void(int64_t)> on_prompt_done;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int stage_worker_loop(StageMemory& memory, core::Verifier& verifier, prefill::Prefill& prefill,
                      core::PoolMultiFn pool, void* user, int64_t chunk, int64_t cache_slots,
                      const std::function<bool(std::string&)>& reset,
                      const std::function<bool(std::string&)>& pool_ok);

} // namespace strata::program
