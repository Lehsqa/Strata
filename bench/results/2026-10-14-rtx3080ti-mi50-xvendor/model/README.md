# IQ3_S, 150,000-token Q4 context: RTX 3080 Ti + MI50

Measured on 2026-10-14 on the machine recorded in the [Phase 0 report](../README.md). The experimental CUDA parent/HIP worker completed a **149,900-token prompt with `--max-context 150000 --kv q4_0`**, returned the synthetic secret at 50% depth, and stopped at EOS. The same resident process also passed 4,096- and 32,768-token requests. This tests an actual near-capacity prompt, rather than only allocating a 150,000-token context.

The original Phase 0 criterion remains p99 < 200 us and remains failed. The user accepted the measured latency for this experimental model test. The full registration-capacity test remains unrun. These results do not establish the remaining production integration gates.

## Configuration and memory

The CUDA parent runs layers `[0,12)` on the RTX 3080 Ti; the HIP worker runs `[12,48)` on the Vega 20 / MI50-class gfx906 card. Split `K=12` is explicit. Each backend has its own executable and runtime, with a shared-memory stage link. Prefill chunks are 1,024 tokens. Generation is greedy, without MTP, lookup drafting, adaptation, prompt/conversation caches or prefix reuse. The Q4 format here is **KV `q4_0`**; the model is IQ3_S, with the pack's native tensor formats retained.

The two native GGUF shards total 77.88 GiB. The logical expert arena is 46.84 GiB, larger than this host's 30.51 GiB RAM. The prepared pack's `dense.bin` is 1.4324 GiB and contains no `experts.bin`; `FileExpertSource` maps the GGUF shards and reads file-backed experts. This run did not load the whole expert arena into RAM or prove the plan's full shared/pinned-memory capacity target.

Automatic cache selection gave the CUDA parent 4,814 slots / 8,688 MiB and the HIP worker 13,348 slots. At READY, the parent reported only 278 MiB free VRAM and printed its low-headroom warning; the worker reported 612 MiB. These successful requests do not establish a sustained-load memory margin. The memory guard remained enabled, with a 2,560 MiB floor shown in both startup logs. `regex` preparation used the project's virtual environment; no global package, driver, power or clock configuration changed.

## Mixed model results

All requests below used one CUDA/HIP process pair and the same 150,000-token Q4 context capacity. Every response had `finish=stop`, eight generated IDs, and `reused=0`. The returned IDs were `[14778, 2149, 44673, 12, 22, 18, 16, 248046]`, corresponding to `amber-falcon-731` followed by EOS. The fixture contains that secret exactly once, at 50% token depth; thinking is disabled by its chat suffix.

| Actual prompt tokens | Engine prefill ms | Engine decode ms | Wall time to first token s | Recall |
|---:|---:|---:|---:|---|
| 4,096 | 30,046.7 | 832.7 | 30.226 | Pass |
| 32,768 | 90,890.5 | 668.4 | 90.961 | Pass |
| 149,900 | 483,676.4 | 606.0 | 483.750 | Pass |

The last prompt leaves 100 tokens of context room, including the requested maximum 40 output tokens. These are single-run observations with uncontrolled file-cache warmth, not sustained throughput measurements. See [completed ladder JSON](logs/mixed-cuda-hip-context-ladder.json), [stdout](logs/mixed-cuda-hip-context-ladder.stdout.log) and [stderr](logs/mixed-cuda-hip-context-ladder.stderr.log).

A separate mixed process pair handled two consecutive 26-token `GEN` requests. Both returned `[19, 248046]` (`4`, EOS), with no reused prefix. Engine prefill/decode times were 2,407.0/718.9 ms and 734.4/138.3 ms. This checks reset between requests; the changed timings also show why file-cache warmth matters. See [short-run JSON](logs/mixed-cuda-hip-short.json).

## Same-GPU control and single-GPU baselines

The CUDA-only stage-link control used two processes on **one physical RTX**, with one resident expert per stage. The one-process reference used two resident experts. The [profile](logs/parity-profile.json) fixes the same resident pairs, `(3,120)` and `(28,288)`, with PCIe misses and lookup drafting disabled. The reference and both remote requests returned `[19,248046]`, `finish=stop`, `reused=0`. This is narrow greedy **token parity** for one short prompt, not numerical/logit parity or a two-card same-vendor comparison. See [reference](logs/remote-reference-cuda-cache2.json) and [remote control](logs/remote-cuda-cuda-cache1.json).

The fixed standalone HIP executable also returned the correct first answer on 4,096-, 32,768- and 131,072-token fixtures, with context capacity 150,000 and Q4 KV. Its CLI prefill counter excludes the final prompt token, reporting 4,095, 32,767 and 131,071. The standalone CLI emitted 40 output IDs, including post-EOS continuation; mixed `--serve` stopped at eight. Their decode timing and configuration differ and should not be treated as a performance comparison.

| Standalone HIP actual prompt tokens | Reported prefill ms | Reported time to first token ms | First answer |
|---:|---:|---:|---|
| 4,096 | 40,577.3 | 41,499.8 | `amber-falcon-731` |
| 32,768 | 187,228.3 | 188,016.7 | `amber-falcon-731` |
| 131,072 | 696,365.2 | 697,157.4 | `amber-falcon-731` |

Short standalone CUDA and fixed HIP runs returned the same first answer `4`. Original baseline logs, including different file-cache states, remain under [logs](logs/); extracted counters are in [results.json](results.json).

## HIP stall and build coverage

The initial gfx906 default standalone run stalled after capture and was stopped at 300 seconds. The traced `STRATA_SH_STREAM=1` control stalled at the eight-token graph launch or immediate stream query, with final trace `staged; launching`, and was stopped at 45 seconds. Setting only `STRATA_SH_STREAM=0`, with debugging off, completed generation. The backend default now avoids that fork stream on gfx906; the fixed default completed the short run and the context ladder. See [initial stall](logs/iq3s-hip-150k-context.log), [trace](logs/iq3s-hip-150k-sh-stream-on-trace.log), [normal stream-off control](logs/iq3s-hip-150k-sh-stream-off-normal.log) and [fixed default](logs/iq3s-hip-150k-default-fixed.log). This diagnosis applies to the tested gfx906/ROCm 6.3.3 configuration.

Both complete model executables compiled and linked: [CUDA sm86](logs/cuda-remote-runtime-build.log) and [HIP gfx906](logs/hip-remote-runtime-build.log). Both model builds used `STRATA_WERROR=OFF`; the earlier Phase 0 CUDA probe's strict build is a separate result. CPU protocol tests passed geometry, malformed commands, sequence/reset lifecycle, crash and timeout checks ([log](logs/remote-protocol-cpu.log)).

The full gfx1100 engine build failed on native ROCm 6.3 hipBLAS compatibility: undeclared `hipblasSetWorkspace` and unmatched `hipblasGemmEx` signatures in the existing RDNA GEMM path ([log](logs/rdna-stage-links-build.log)). This differs from the earlier successful gfx1100 **probe-only** build. No RDNA GPU execution or SYCL build was performed.

## Reproduction and evidence

The source is commit `40bcc2ca4f971902559f3794273ca2946748355e` plus the uncommitted experimental stage-link/gfx906 fixes. [results.json](results.json) records source hashes, complete run arguments and measured counters. The executable's `INFO` confirms `context=150000`, `kv=q4_0`, and disabled reuse caches. No nonfinite failure was reported by these runs; this is not comprehensive finite-logits validation.

Generate the deterministic fixture files from the repository root, using its prepared Python environment:

```sh
.venv/bin/python bench/results/2026-10-14-rtx3080ti-mi50-xvendor/model/generate_context_fixtures.py \
  --gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --out bench-local/iq3s-context-fixtures
```

The portable generator uses `tools/ab_engine.py::long_prompt`, seed 7 and the recorded chat wrapper. [fixture_manifest.json](fixture_manifest.json) retains exact text/token-file hashes, counts, needle position and tokenizer roundtrip checks from the measured run. Token-file hashes include the file's trailing newline; runner token hashes use canonical comma-separated IDs without that newline. The original generator is retained for provenance in [logs](logs/generate_iq3s_context_fixtures.py). The portable generator's syntax and `--help` were checked; its full reproduction has not been rerun.

Run [run_model.py](run_model.py) using the arguments recorded in each completed JSON. For the mixed ladder, supply the CUDA executable, `--worker` pointing to the HIP executable, `--worker-backend hip --split 12 --context 150000 --kv q4_0 --prefill 1024 --cache auto`, and repeat `--tokens-file` for 4,096, 32,768 and 149,900 tokens. The recorded run additionally sets `--pcie-frac 0 --lookup-chain 0` and `STRATA_ROUTE_TAIL_SKIP=0`. It uses `.venv`, the native pack and both GGUF shards. The HIP child strips CUDA visibility and uses HIP device 0. The runner starts the engine's local stdio protocol; it does not expose a network service.

Published logs replace home/repository paths with `<TARGET_HOME>`/`<TARGET_REPO>`. Three large synthetic CLI prompt-ID lines are omitted and replaced by a fixture-hash reference; all other output is retained. [log manifest](logs/manifest.json) records both original and published file hashes. No prompt token list or model data is included.

## Actual engine fault tests

Two fresh mixed process pairs used the same model arguments and 4,096-token fixture, with context 150,000 and Q4 KV. The driver injected each fault five seconds after `GEN`, during prefill; it targeted only its own process IDs.

| Injected fault | Observed result | Time after injection | Cleanup assistance |
|---|---|---:|---|
| SIGKILL HIP worker | Parent reported `ERR remote stage: worker exited before replying`, exited 1; worker absent | 0.991285 s | None |
| SIGKILL CUDA parent | Parent exited -9; HIP worker reached dead/zombie state through parent-death handling | 0.585565 s | None |

Neither case returned output tokens or `DONE`. A later `/proc` check confirmed both parent and worker PIDs absent in both cases. See [fault results](logs/engine-fault-results.json), [no-orphan follow-up](logs/engine-fault-no-orphans.json), per-case logs and [driver](logs/engine_fault_validation.py). These establish bounded failure/cleanup for two prefill injections, not restart/resume recovery, a decode-phase crash test, or an actual hung-worker timeout test. CPU protocol timeout coverage remains a separate result.

No model quality sweep, sampling, MTP, checkpoint, Python-server/setup, automatic split, alternative card-order, full registration capacity or sustained throughput test is claimed.
