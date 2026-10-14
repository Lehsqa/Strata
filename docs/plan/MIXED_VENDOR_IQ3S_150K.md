# IQ3_S, 150K context, RTX 3080 Ti + MI50

Requested on 2026-10-14, after the user accepted the measured Phase 0
handoff latency. The original latency threshold remains in the benchmark
report; accepting it for this experiment does not change that measurement.

## Target and memory constraint

The target has an RTX 3080 Ti (12 GiB), a Vega 20 / MI50-class card
(31.98 GiB reported), and 30.51 GiB system RAM. Existing IQ3_S GGUF shards
total 83,617,662,656 bytes (77.88 GiB). Their routed experts require
50,292,326,400 bytes (46.84 GiB) in the engine's resident arena. That arena
does not fit this host, even before sessions and CPU scratch.

Use the existing native GGUF file tier: make the small dense/index/tokenizer
pack, leave the GGUF files in place, and map experts read-only from their
original shards. Both processes can share the operating system's file cache.
This extends the original mixed-worker plan, which refused `--mmap-experts`
in its first version. It does not establish the original full-capacity
registration gate.

## First executable test path

Implement the existing two-process stage design, with CUDA main and HIP last
stage, an explicit layer boundary, and static expert caches. Use the engine's
stdin `--serve` protocol directly. Allocate 150,000 context cells with
`--kv q4_0`. Start with greedy decoding, no MTP, no prompt cache, and
synchronous prompt-chunk forwarding. Preserve the existing local stage paths
when no remote stage is requested.

The main and worker each load their own dense layer range, session range,
expert cache, and CPU pool. A versioned memfd carries commands, decode
handoffs, prompt rows, picks, and errors. Validate geometry and model identity
before inference. Keep CUDA visibility out of the HIP worker environment,
keep ROCm library paths local to that worker, and detect worker death while
waiting. Unsupported remote options must fail clearly.

Resident-complement helpers, adaptive caches, router lookahead, batch paths,
vision, session files, and pipelined verify windows stay outside this initial
test path. Their current assumptions require further integration.

## Checks, in order

1. Build the current full CUDA and gfx906 engines on the target; these are
   distinct from the Phase 0 probes.
2. Finish the native pack and tokenizer without copying all expert tensors.
3. Run a bounded single-vendor IQ3_S test at 150,000 configured context cells
   with Q4 KV. Record actual allocations, startup errors, and finite output.
4. Check the new transport protocol on the CPU, including malformed commands,
   reply ownership, bounds, and worker termination.
5. Check remote CUDA-to-CUDA inference against the existing local engine.
6. Run the same model on HIP alone, then mixed CUDA-to-HIP with a short prompt.
   Record the actual layer boundary and allocations on both GPUs.
7. Increase prompt length in bounded steps. A short prompt with 150K context
   allocated is a capacity test; a 150K-token prompt is a separate inference
   test and must be reported separately.

No successful mixed-model or long-prompt result is implied by this plan.


## Measured outcome (2026-10-14)

The experimental CUDA parent/HIP worker completed IQ3_S recall requests with
4,096, 32,768, and 149,900 prompt tokens, each with 150,000 Q4 context cells.
The final prompt returned the correct needle and EOS after 483.7 seconds of
prefill (309.92 tokens/s). Worker and parent termination tests also passed.
See the [model report](../../bench/results/2026-10-14-rtx3080ti-mi50-xvendor/model/README.md)
for commands, source hashes, memory headroom, and remaining validation gaps.
This result covers the restricted greedy CLI lane described above, not the
original plan's MTP, cache, automatic split, or server/setup phases.


## Post-test verification and handoff

After the mixed-model run, the following work completed:

- Verified the saved 4,096-, 32,768-, and 149,900-token results against exact
  expected token IDs, EOS, zero prefix reuse, Q4 KV, and process exit status.
- Tested actual engine failures during 4K prefill. Killing the HIP worker made
  the CUDA parent report an error and exit in 0.991285 seconds. Killing the
  parent ended the worker within 0.585565 seconds; a later check found no
  surviving test processes.
- Rebuilt the full CUDA sm86 and HIP gfx906 engine targets. Reran the CPU
  stage protocol test through CMake/ctest and a strict standalone C++ build,
  plus the Phase 0 shared-memory, CLI, and Python launcher tests.
- Published sanitized model and fault-test logs, fixture metadata, source
  hashes, and reproduction instructions. Checked evidence hashes, document
  links, path redaction, and source/report diff whitespace. Raw evidence logs
  retain their recorded formatting and hashes.
- Updated AMD, multi-GPU, and probe documentation to describe the tested
  experimental CLI lane. Synced the source and report bundle to the target PC.

The runtime fixes verified by these runs include disabling the shared-expert
stream fork by default on gfx906 after a reproduced stall, preserving expert
residency-table initialization in remote mode, transporting per-row nonfinite
status, and validating the full model geometry and split before inference.
The mixed stage path remains opt-in; the gfx906 stream default is a measured
backend fix. CUDA's stream default is unchanged.

The original p99 < 200 microsecond handoff gate remains failed, and the full
registration-capacity gate remains unmeasured. Full RDNA engine compilation
is blocked by existing ROCm 6.3 hipBLAS API incompatibilities; SYCL was not
built. MTP, sampling, caches, automatic splitting, and Python server/setup
integration remain outside this tested lane. Primary GPU free memory was
only 278 MiB at READY, so these runs do not establish sustained-load or OOM
headroom. The portable fixture generator was syntax/help checked; its full
reproduction was not rerun.

The commit accompanying this handoff includes all implementation changes and
both the Phase 0 and model-test evidence bundles. The reports describe the
source state at measurement time; their source hashes identify the tested
engine files independently of the later packaging commit.
