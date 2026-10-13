# Plan: NVIDIA RTX + AMD Instinct MI50 (gfx906) in one layer split ("mixed build")

## Context

Strata runs one model across several cards with a **layer split**: contiguous layer ranges per GPU, one hand-off per
verify window through pinned host RAM (no P2P). Today this works only within one vendor, because the backend is
chosen at **compile time**:

- `CMakeLists.txt:88-89` fails if `STRATA_ENABLE_CUDA` and `STRATA_ENABLE_HIP` are both on. gfx906 is its own
  build: `STRATA_HIP_GFX906` (:64-69) makes fake `CUDA::cudart` targets on top of HIP, and
  `include/strata/platform/hip_compat/strata_hip.h` `#define`s every `cuda*` name to its `hip*` name.
- All libraries are static and both builds produce the same C++ symbols. Host code has about 3,000 `cuda*` calls
  (`generate.cpp`, `verify.cpp`, `prefill.cpp`, `session.cpp`, `mtp.cpp`, ...). Stage 0 is wired to device 0 of the
  single runtime.
- Docs say mixing vendors is not supported (`docs/MULTI_GPU.md:53`, `docs/AMD_HIP.md:9-10`). setup.py makes you
  pick one vendor (`setup.py:5050-5064`), and setup cannot build gfx906 at all.

**Goal:** an NVIDIA RTX card plus an MI50/MI60 run Qwen3.8-Flash-Next together, with the same experience as two
NVIDIA cards (`--gpus` lists both, `layer_split auto`). Linux first.

**Chosen approach (user decision):** a stage worker, done in steps.
- Step 1: the last stage runs in a second `strata` binary built for the other vendor. The engine that drives it
  talks to it through a shared-memory command ring with a doorbell.
- Step 2 (optional): the same worker, loaded in-process as a `.so` with hidden symbols, like llama.cpp loads its
  backends.

The existing hand-off already goes through host memory and is driven by the host thread, so this needs no P2P and
no change to the kernels.

## Key design decisions

1. **The main process owns stage 0 and the MTP drafter. The worker owns the last stage and the target head.**
   - Stage 0 is the top-level objects (`wt`, `ss`, `ver`, `sp`, `xcache`, `drive`) used throughout
     `generate.cpp`, so making it remote is impractical.
   - With no in-process later stage, `last_st == nullptr`, so the existing single-GPU MTP branches are taken
     (`generate.cpp:4499-4504` `OnDevice on_mtp(last_st ? ... : -1)`).
   - The main process also loads the head for the drafter: the condition at `generate.cpp:4542`
     (`!multi_gpu`) already holds. That costs about 0.5-0.8 GB of VRAM on card 0.
   - The drafter needs the final R rows (≤320 KB per window) and the prompt rows (40 KB per token) back from the
     worker. It reads them zero-copy through mapped host memory.
2. **The vendor of each process follows card order.** The first card's vendor picks the main exe, the last card's
   vendor picks the worker exe, and both directions work. The server keeps its rule of putting the faster card
   last. The default order is picked only after measuring RTX→MI50 against MI50→RTX (AGENTS.md: defaults change
   only with numbers).
3. **Each process loads only its own layers' experts into its own arena.** Total RAM equals one arena, each driver
   registers its own pages, and transparent huge pages stay. `ArenaExpertSource::open` gets an optional
   `layer_lb, layer_le` (default: all layers). The full virtual range is reserved so `blob(l,e)` offsets stay
   global, and only that range is loaded and registered. The per-layer slice registration already exists in
   `src/core/pinned.cu:394-483`.
4. **Separate processes also separate the per-architecture global latches** (e.g. `g_qfuse_hc` at
   `verify.cpp:141`). Each process latches for its own card.
5. **Opt-in only.** The new flags are `--stage-remote cuda:EXE|hip:EXE`, `--stage-remote-devices N` and the
   internal `--stage-worker fd:N`. Without them every new branch is dead code and the default output stays
   byte-identical.

## Phase 0: feasibility spike (1-2 weeks, go/no-go)

New `tests/xvendor/`, behind a CMake option `STRATA_XVENDOR_TESTS` (default OFF). It builds in the CUDA tree
`build/` and the gfx906 tree `build-906/`. It is not part of ctest, because it needs both cards.

- `shm_region.hpp`: memfd + mmap, header with atomic sequence words, futex wait/wake across processes (spin, then
  `FUTEX_WAIT`).
- `xvendor_handoff.cu` (`--role main|worker --fd N`):
  1. Host ping-pong latency: p50/p99, spin-only and with futex.
  2. Emulated decode hand-off:
     - The same memfd range is registered with `cudaHostRegister(Portable|Mapped)` in one process and
       `hipHostRegister(Mapped)` in the other.
     - Writes and reads use `strata::kernels::copy_from_mapped` plus graph replay, following
       `tests/hip/handoff.cpp`.
     - Payloads are 8×12804 floats out and 8×10240 floats plus 8 int32 back, with a stale-payload check.
     - **Pass:** p99 < 0.2 ms against a 40-55 ms window, and no stale payload in 10⁴ rounds.
  3. Cross-process GPU doorbell (`doorbell_wait` on a flag in the region). This gates only a later pipelined mode.
  4. A 32 GiB registration in both processes, plus a 16 GiB anonymous arena in each at the same time. This
     checks memlock and `/dev/shm` limits and H2D bandwidth.
- `xvendor_inproc.cpp` (for Step 2): `dlopen` `libxprobe_cuda.so` and `libxprobe_hip.so`, both built with
  `-fvisibility=hidden -Wl,--exclude-libs,ALL -Wl,-Bsymbolic`. Test init in both orders, one buffer registered in
  both runtimes, teardown order, and device visibility (HIP honours `CUDA_VISIBLE_DEVICES`).
- **Co-install notes** (new section in docs/AMD_HIP.md):
  - Build the gfx906 worker inside the community ROCm 7.14 container.
  - Copy the binary and its ROCm runtime closure out, using setup.py's existing `hip_runtime_closure()` and
    `hip_runtime_beside_exe()`. Run it natively on the in-kernel amdgpu driver next to the NVIDIA driver; no
    amdgpu-dkms.
  - The user needs the `render`/`video` groups.
  - Set ROCm's `LD_LIBRARY_PATH` only in the worker's environment.
  - A HIP process must never inherit `CUDA_VISIBLE_DEVICES`.

## Phase 1: protocol + remote stage on one vendor (3-4 weeks)

### 1a. Interfaces. New code; the edits to shared files are small.

- **New `include/strata/core/stage_link.hpp`:** an abstract `StageLink` with the calls the Verifier chain makes
  on `next_` today:
  - `run`, `commit` (posted), `wait_commit`, `set_logit_bias`;
  - `set_sampling`, `set_history`, `set_spec_q`, `set_head_sampling`;
  - `copy_logits`, `vocab`, `nonfinite_row/pos/in`, `final_R_all`.
- **`include/strata/core/verify.hpp` + `src/core/verify.cpp`:**
  - add `StageLink* link_ = nullptr`, `set_next_link()` and `has_next()`;
  - add an `if (link_)` branch next to each `next_` forwarder: :522, :2307, :2440, :2529, :2539, :3158 and the
    inline ones in verify.hpp;
  - use `has_next()` at :2513 and :2765;
  - leave the batch paths (:2874, :2938, :2990) as they are, since `--batch` is refused.
- **`include/strata/prefill/prefill.hpp` + `src/prefill/prefill.cpp`:**
  - add an abstract `PrefillLink` with `chunk_slot(b)` and a blocking `run_chunk(...)`;
  - add `set_stage_link()`, and a public receiver entry `run_stage_chunk()` that wraps the private `run_impl`
    and `hand_in_`;
  - the sender at :4050-4070 sends to the link when one is set;
  - skip allocating `m.hand` at :1078 when there is a link.
  - SYCL uses the root `prefill.hpp`, so the SYCL build must be checked.

### 1b. Wire protocol: new `include/strata/program/stage_link_proto.hpp` (header-only, testable on CPU)

- **One memfd region:**
  - a 4 KiB header: magic `STRATA-STAGE-V1`, a geometry check (n_embd, hc, n_layers, kVerifyMaxT,
    `handoff_floats` from `verify.hpp:177`), `cmd_seq`/`done_seq`, heartbeat, progress text, `Cmd`/`Reply`;
  - a command payload area and a reply payload area;
  - the decode hand-off area, `kVerifyMaxT*handoff_floats*4`, registered on both sides;
  - the final-R area;
  - two prompt slots and a rows-back area, each `chunk*hc*n_embd*4`.
- **Ops:**
  - `HELLO`: the worker reports backend, device, arch, CUs/SMs, clock, VRAM, measured PCIe GB/s
    (`probe_pcie_h2d_gbps`), version/hash and geometry.
  - `ASSIGN` → `READY`: lb, le, max_t, chunk cap, head. The worker reports the chunk it chose, its cache slots,
    the VRAM left, n_vocab and the checkpoint part size.
  - Per window and request: `RUN`, `COMMIT` (posted, one deep), `WAIT_COMMIT`, `SET_SAMPLING`,
    `SET_HEAD_SAMPLING`, `SET_LOGIT_BIAS`, `COPY_LOGITS`.
  - Prompt and checkpoints: `PROMPT_CHUNK`, `CKPT_SAVE` / `CKPT_RESTORE` (serialize `ConversationCheckpoint`
    from `include/strata/core/conversation_cache.hpp`).
  - Housekeeping: `STATS`, `RELEASE_GPU_WAITS`, `SHUTDOWN`.

### 1c. Main side: new `src/program/remote_stage.{hpp,cpp}`, implementing `StageLink` and `PrefillLink`

- **Spawn:** `posix_spawn` of `{exe, --stage-worker fd:N, <main argv minus --stage-remote*>}`. The memfd is
  inherited.
- **Environment per vendor:**
  - a HIP worker gets `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER` removed and `HIP_VISIBLE_DEVICES=<dev>` set;
  - a CUDA worker gets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<dev>`, with
    `HIP_VISIBLE_DEVICES` removed;
  - `LD_LIBRARY_PATH` comes from `STRATA_STAGE_LD_LIBRARY_PATH`.
- **Waiting:** spin for `STRATA_STAGE_SPIN_US` (default 50 µs), then futex in 100 ms slices. Each slice checks
  whether the worker exited (pidfd/waitpid), whether it reported FAILED, and how old its heartbeat is. The
  progress text feeds the existing stall watchdog.
- **Worker crash:** main prints one message, `strata serve: the remote stage (pid P, <exe> on HIP0 gfx906)
  exited ...; its last step: "..."`, and exits non-zero. The server already restarts a dead engine.

### 1d. Worker: new `src/program/stage_worker.cpp`, dispatched from `generate.cpp main()` right after option parsing

The worker is a trimmed copy of the in-process later-stage setup (`GpuStage`, `generate.cpp:1057-1077`,
:3433-3532, :7461-7556). It reuses:

- `apply_arch_defaults` for its own card (:2190-2215);
- `load_stage_weights` and `add_foreign` (:2992, :3437-3471), with weights always trimmed to its range;
- `NativeDense` and `NativeHead`;
- `session_bytes`/`session_init` for [lb, le);
- `ArenaExpertSource` limited to its range;
- an `ExpertCache` sized by the `stage_room` rule (:5305-5370);
- its own `host_res`/`d_res`, `ExpertPool` and `Drive` (`drive_pool_multi`, :5548-5640 subset), with its own
  PCIe probe;
- a `Verifier` with `set_stage(lb, -1, mapped hand-off, nullptr)`;
- a `Prefill` with `set_stage(lb, -1, nullptr)`, whose `on_chunk` copies the rows into rows-back.

It also:

- calls `prctl(PR_SET_PDEATHSIG, SIGKILL)` so it dies with main;
- logs only to stderr, prefixed `strata worker (HIP0 gfx906):`;
- takes split-wide values from main as explicit args/env: route tail skip / max-cached (:8286-8300), kv mode,
  `--pcie-frac`.

### 1e. `generate.cpp` touchpoints (add `RemoteStage* remote`, `split_any = multi_gpu || remote`; `stages` stays empty)

1. Options, parse and validation: :597-602, :2042, :2299-2347, the usage text near :911, and the `--stage-worker`
   dispatch.
2. Spawn and HELLO before stage creation (:3433), after the split-skip decision (:3348-3376).
3. Switches that apply to any split:
   - `set_commit_async(!split_any)` :3390;
   - `stage_trim` :2982;
   - `hi0` session range :4265, :4436;
   - token graph off :6237 (critical);
   - single-GPU prefill features off: :4273, :6952, :6985, :8215, :11155;
   - `pf_parts` :7037;
   - adaptive overlap off :8358.
4. Arena range `[0, K)` at :4768.
5. Verify wiring at :7461-7556: `ver.set_stage(0, K, nullptr, remote->handoff_dev())`, `ver.set_next_link(remote)`,
   and wait for READY before `ver.init` (:7609).
6. Prompt path:
   - :7284 `sp.set_stage(0, K, nullptr); sp.set_stage_link(remote)`;
   - chunk = min(main, READY) at :7290-7367;
   - skip `STRATA_PREFILL_HELP` at :7377.
7. Phase 1 uses a manual `--layer-split K` only.

## Phase 2: checkpoints, auto split, robustness (1.5-2 weeks)

- **Prompt-cache checkpoints:**
  - The worker's part travels as bytes and is stored in main's `ConvCheckpoint::stage_parts`, so LRU and
    accounting stay the same.
  - `parts.size() == n_later + 1` at :8117 and :8227 (`n_later` counts the remote stage).
  - `on_chunk` remote branch at :8239-8258.
  - `checkpoint_at` at :8126.
  - Restores at :9354-9398 and :10578-10610.
- **`--layer-split auto`:**
  - The remote stage's inputs (free VRAM → `stage_room`, CUs×GHz, link GB/s) come from HELLO at :3694-3728,
    :3926, :4132.
  - The drafter reserve (`kDrafterMib`) and the head's bytes count against stage 0.
  - The profile split happens at :4218-4240; main then sends ASSIGN.
- **Stats:** worker slots/MiB in the INFO totals (:9067), a "remote:" timing line (:9491-9644),
  `release_gpu_waits` (:9151, :11398), SHUTDOWN on exit.
- **`--split-skip-if-fits`:** when main decides it fits, it shuts the worker down and runs alone.

## Phase 3: RTX + MI50 bring-up (1-2 weeks)

- Run the gfx906 worker next to the CUDA main and the reverse, and measure both card orders.
- **Two CPU pools contend:** a parked pool spins for `kSpinBeforeSleep = 20 ms`
  (`include/strata/kernels/cpu/pool.hpp:216`). Tune `STRATA_POOL_SPIN_US`, or add `ExpertPool::park_now()`
  called while a process waits on the other one.
- **Docs:** new section in docs/MULTI_GPU.md, "NVIDIA + AMD in one split (experimental, Linux)", and a gfx906
  co-install section in docs/AMD_HIP.md. Both carry measured numbers and say what they were measured on.

## Phase 4: setup and server (1.5-2 weeks)

- **`serve/server.py`:**
  - config `"gpu": ["cuda:0","hip:0"]`, `"backend": "mixed"`, `"exes": {"cuda":..., "hip":...}`, lib dirs per
    backend, `"gpu_scores"`;
  - new `gpu_devices(cfg)`; `gpu_list()` (:2308) keeps returning integers for one vendor;
  - `ordered_gpus` (:2587) uses `gpu_scores` with the same 5% rule and VRAM guard;
  - `engine_args` (:2407) adds `--stage-remote`/`--stage-remote-devices`;
  - `child_env` (:2622) sets only the first card's vendor variable, plus `STRATA_STAGE_LD_LIBRARY_PATH`;
  - the exe started at :6406 is the one for the first card's vendor.
- **`setup.py`:**
  - On Linux with one NVIDIA and one AMD card, add a third choice at :5050-5064: "both together
    (experimental)". Also add `--backend mixed`.
  - Let `amd_problem()` accept gfx906 for that choice.
  - New `build_engine_gfx906()`, following `build_engine_hip` at :2632: `build-906/`, `engine-906/`, BUILD.json
    `"backend": "hip-gfx906"`, runtime closure beside the exe. If the ROCm found lacks gfx906 rocBLAS kernels, it
    prints the container recipe.
  - Build both engines.
  - Get comparable speed scores from a new `strata-device --probe-json` (`src/core/device_main.cpp`: CUs/SMs,
    clock, VRAM, a measured device-bandwidth copy).
- **New `tools/test_setup_mixed.py`** (mocked nvidia-smi + KFD gfx906, following `tools/test_setup_amd.py`), plus
  a golden entry in `tools/test_setup_golden.json`.

## Step 2 (optional, after Step 1 numbers): worker in the same process (2-3 weeks)

- In `build-906/`, a new option `STRATA_STAGE_SO=ON` builds `libstrata_stage_hip.so`:
  - PIC static libraries, `-fvisibility=hidden`;
  - `-Wl,--exclude-libs,ALL -Wl,-Bsymbolic` and a version script;
  - it exports only `extern "C" strata_stage_entry_v1(const StrataStageArgs*)`.
- Main gets `--stage-inproc LIB`: `dlopen(RTLD_NOW|RTLD_LOCAL)`, then the worker loop on a `std::thread` over an
  anonymous-mmap region with the same protocol. It never calls `dlclose`.
- **Blockers:** two runtimes in one process (Phase 0 `xvendor_inproc`), device visibility, a worker crash now
  killing the engine, and two build trees still needed.
- **Gain:** a few µs per window against 40-55 ms windows. Do it only if Step 1's measurements or deployment
  problems call for it.

## v1 scope

**Supported:**
- two stages, with the remote stage last;
- decode with `--mtp`/`--spec`, greedy and sampled, penalties, logit bias;
- prompt chunks, pipelined;
- prompt-cache checkpoints;
- manual K, then `auto`.

**Refused with a one-line message:**
- 3+ stages;
- `--pipeline-windows`, `--batch`/`--batch-mtp`;
- `STRATA_PREFILL_HELP` (switched off);
- `--peer-device`, `--expert-cache-device*`, `--remote-expert-opt`, `--vram-elastic`;
- conversation parking / session files, `--vision`;
- `--mmap-experts`/`--resident-experts`/`--shared-expert-arena`;
- `window_logprobs`/`STRATA_LOGPOS`;
- Windows.

The worker's expert cache stays static in v1, and the log says so.

## Critical files

- Changed: `src/program/generate.cpp`, `src/core/verify.cpp` + `include/strata/core/verify.hpp`,
  `src/prefill/prefill.cpp` + `include/strata/prefill/prefill.hpp`, `src/core/expert_source.cpp` +
  `src/core/pinned.cu`, `CMakeLists.txt`, `serve/server.py`, `setup.py`, `src/core/device_main.cpp`.
- New: `include/strata/core/stage_link.hpp`, `include/strata/program/stage_link_proto.hpp`,
  `src/program/remote_stage.{hpp,cpp}`, `src/program/stage_worker.cpp`,
  `tests/program/stage_link_proto_test.cpp`, `tests/xvendor/*`, `tools/test_setup_mixed.py`.

## Verification

1. **Default path unchanged:**
   - build CUDA (`build/`), RDNA HIP (`build-hip/`), gfx906 (`build-906/`) and SYCL;
   - run all ctest suites;
   - greedy text before and after with `tools/split_mtp_batch_parity.py dump/compare`, on one GPU and on an
     in-process 2-GPU split;
   - `tools/conversation_cache_parity.py`;
   - `tools/ab_engine.py` speed within noise;
   - `STRATA_VERIFY_ARENA=1` checksum unchanged.
2. **Protocol test on CPU only:** ctest `stage_link_proto_test`. It covers offset math against `handoff_floats`,
   a Cmd/Reply round trip over threads and then over `fork()`+memfd, futex timeout and peer death, checkpoint
   serialization, and posted-COMMIT ordering.
3. **Bit-exact on one vendor:**
   - remote stage on the same card against an unsplit run;
   - CUDA main + CUDA worker on two NVIDIA cards against the in-process split with the same K and equal caches
     (test knob `STRATA_STAGE_CACHE_SLOTS`, plus `--pcie-frac 0`, `STRATA_ROUTE_TAIL_SKIP=0`).
   - Greedy text must be identical, and draft acceptance within ±3 points.
   - The same check on 2× MI50 (gfx906 main + gfx906 worker).
4. **RTX + MI50:**
   - token agreement against each single-vendor run;
   - the NaN guard never fires;
   - draft acceptance;
   - decode and prompt tok/s per docs/COMMUNITY_BENCHMARKS.md, against same-day baselines: RTX alone, MI50
     alone, and both card orders.
5. **Fault tests:** `kill -9` the worker mid-window and mid-chunk; main must exit with the message and the
   server must restart it. Killing main must kill the worker.

## Effort and risk

| Phase | Effort | Risk |
|---|---|---|
| 0 spike | 1-2 wk | High but cheap: gfx906 ROCm next to the NVIDIA driver, `hipHostRegister` coherence, memlock limits |
| 1 protocol + remote on one vendor | 3-4 wk | Medium: the worker's copied setup can drift from `generate.cpp`; parity tests catch it |
| 2 checkpoints, auto, robustness | 1.5-2 wk | Medium |
| 3 RTX + MI50 bring-up | 1-2 wk | High for speed: two CPU pools, drafter on card 0, gfx906 maturity |
| 4 setup/server | 1.5-2 wk | Medium |
| Step 2 in-process `.so` | 2-3 wk | High, small gain |
