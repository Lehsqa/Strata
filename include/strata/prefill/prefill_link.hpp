#pragma once
#include <cstdint>
#include <string>

namespace strata::prefill {

/// A next prompt stage. Both host hand-off slots are owned by the link and remain valid until drain.
/// run_chunk blocks until the successor has consumed its input rows; single_chunk describes the whole
/// prompt, not just this chunk, so helper decisions match the local layer split.
class PrefillLink {
public:
    virtual ~PrefillLink() = default;
    virtual float* chunk_slot(int slot) = 0;
    virtual bool run_chunk(const int64_t* tokens, int64_t n, int64_t pos0, const float* host_rows,
                           bool single_chunk, std::string& err) = 0;
    virtual bool drain(std::string& err) = 0;
};

} // namespace strata::prefill
