#pragma once
#include "strata/kernels/sampler.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                            float* out, int64_t layer);

/// A serialized next stage. Pointers and callbacks stay in the caller's process; remote implementations
/// transport their contents, never their addresses. Batch and pipelined windows are not part of v1.
class StageLink {
public:
    virtual ~StageLink() = default;
    virtual bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user,
                     int32_t* out, std::string& err) = 0;
    virtual bool commit(int n_keep, std::string& err) = 0;
    virtual bool wait_commit(std::string& err) = 0;
    virtual bool set_logit_bias(const std::vector<float>& bias, std::string& err) = 0;
    /// sp.logit_bias is caller-device memory; remote bias values arrive through set_logit_bias.
    virtual void set_sampling(const strata::kernels::SamplerParams& sp) = 0;
    /// history is a caller-device buffer. A remote adapter must stage its rows to host before sending.
    virtual void set_history(const int32_t* history, int history_len) = 0;
    /// q is host memory, valid until the following run completes.
    virtual void set_spec_q(const int32_t* q, int n_q) = 0;
    virtual void set_head_sampling(bool on) = 0;
    virtual bool copy_logits(int t, float* host) const = 0;
    virtual int64_t vocab() const = 0;
    virtual int nonfinite_row() const = 0;
    virtual int64_t nonfinite_pos() const = 0;
    virtual bool nonfinite_in(int t) const = 0;
    /// Remote implementations expose caller-device rows, valid until the following run.
    virtual const float* final_R_all() const = 0;
};

} // namespace strata::core
