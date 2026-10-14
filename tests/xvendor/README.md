# Cross-vendor Phase 0 probes (Linux, manual)

These opt-in probes test whether a CUDA process and a HIP process can register the
same memfd pages and exchange mapped payloads. They also test runtime coexistence
in a single process. They do not add mixed-vendor inference to Strata. No probe is
registered with ctest, and `STRATA_XVENDOR_TESTS` defaults to `OFF`.

Status (2026-10-14): actual CUDA 12.0 and gfx906 ROCm 6.3.3 builds, mixed
payload/doorbell checks, limited registration and narrow runtime coexistence
passed on an RTX 3080 Ti + Vega 20 host. Serialized p99 still fails the 200 us
gate; full 32 + 16 + 16 GiB capacity was not tested on its 30.51 GiB host RAM.
See the [hardware report](../../bench/results/2026-10-14-rtx3080ti-mi50-xvendor/README.md).
RDNA gfx1100 also compiled and linked with ROCm 6.3.3; runtime checks apply to
gfx906 only. These probes add no engine inference support. Other toolchain
versions remain unverified. After the user accepted the measured latency,
a separate experimental CLI stage implementation passed IQ3_S recall with a
149,900-token prompt and 150,000 Q4 context cells on RTX 3080 Ti + MI50. See
[model report](../../bench/results/2026-10-14-rtx3080ti-mi50-xvendor/model/README.md)
for scope and remaining gates. The successful gfx1100 build above is probe-only;
the full engine build failed on existing ROCm 6.3 hipBLAS API incompatibilities.

## CPU checks (no GPU SDK needed)

```sh
cmake -S tests/xvendor -B /tmp/xvendor-cpu -DSTRATA_WERROR=ON
cmake --build /tmp/xvendor-cpu -j
/tmp/xvendor-cpu/strata-xvendor-shm-test
/tmp/xvendor-cpu/strata-xvendor-cli-test
python3 tests/xvendor/test_runner.py
# Expected nonzero failures; these exercise loader validation without GPU libraries:
/tmp/xvendor-cpu/strata-xvendor-inproc --cuda /missing.so --hip /missing.so
/tmp/xvendor-cpu/strata-xvendor-inproc --cuda /missing.so --hip /missing.so --order bogus
```

The same CPU targets are available from the parent build with
`-DSTRATA_XVENDOR_TESTS=ON`, leaving both GPU backends off. SYCL and non-Linux
configurations explicitly refuse this option.

## Build each vendor separately

Use your NVIDIA architecture and the CUDA version recommended in the main build
docs; `86` below is an RTX 30 example. The probe library links the shared runtime
only, with hidden symbols and a narrow C ABI; it does not link Strata's engine.

```sh
cmake -S . -B /tmp/xvendor-cuda -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_XVENDOR_TESTS=ON
cmake --build /tmp/xvendor-cuda --target strata-xvendor-handoff xprobe_cuda -j
```

For MI50/MI60/Radeon VII, use the community image
`mixa3607/rocm-gfx906:7.14-complete`, as described in
[AMD_HIP.md](../../docs/AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source).
Build inside that image, with the source mounted at `/workspace/Strata`:

```sh
cmake -S /workspace/Strata -B /workspace/xvendor-hip \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_XVENDOR_TESTS=ON
cmake --build /workspace/xvendor-hip --target strata-xvendor-handoff xprobe_hip -j
```

RDNA uses a separate build with `-DSTRATA_ENABLE_HIP=ON` and the actual
`-DCMAKE_HIP_ARCHITECTURES=gfx1100` (example). The handoff source is explicitly
compiled as HIP on either AMD backend. CUDA 12.0 and gfx906 ROCm 6.3.3 builds
passed on the reported host. RDNA gfx1100 also compiled and linked with ROCm
6.3.3; its hardware execution and other toolchain versions remain unverified.

Run the copied HIP executable on the native host with the matching image's ROCm
runtime tree; keep rocBLAS/Tensile files and relative paths intact. A read-only
`ldd` check helps identify missing ELF libraries. Setup's
`hip_runtime_closure()` and `hip_runtime_beside_exe()` copy **Windows DLLs**, not
Linux `.so` closures. They cannot package this Linux probe. See the co-install
notes in [AMD_HIP.md](../../docs/AMD_HIP.md#linux-cuda-and-hip-probe-co-install).

## Two processes, both vendor roles

Set `CUDA_EXE` and `HIP_EXE` to the two built executables. If the matching HIP
runtime needs a library path, pass its directories with `--hip-libdir`; this is
applied to the HIP child only. The launcher removes CUDA visibility variables
from the HIP child because HIP also honors `CUDA_VISIBLE_DEVICES`. It removes
HIP/ROCR visibility variables from the CUDA child, uses CUDA PCI bus order, and
passes the **same** memfd to both fresh executions.

```sh
CUDA_EXE=/tmp/xvendor-cuda/tests/xvendor/strata-xvendor-handoff
HIP_EXE=/opt/strata-xvendor/hip/strata-xvendor-handoff
ROCM_LIB=/opt/strata-rocm/lib:/opt/strata-rocm/lib64
for main in cuda hip; do
  for mode in spin futex doorbell; do
    python3 tests/xvendor/run.py --cuda "$CUDA_EXE" --hip "$HIP_EXE" \
      --main "$main" --mode "$mode" --rounds 10000 \
      --cuda-device 0 --hip-device 0 --hip-libdir "$ROCM_LIB" --timeout 180
  done
done
```

`spin` uses CPU polling; `futex` spins briefly then sleeps. Each round sends exactly
`8 x 12804` FP32 values and receives `8 x 10240` FP32 values plus eight int32
indices. Graphs are captured once and replayed with changing payloads. The worker
checks every input on its GPU and the main process checks every return/index both
in mapped memory (doorbell mode) and after copying into its GPU.

The serialized modes report round-trip p50/p99 wall time, excluding the main's
host fill/upload and final readback/check but including both graph directions,
CPU rendezvous and the worker's full processing/check. Report `p99_lt_200=yes`
and zero stale values over at least 10000 rounds in **both** vendor directions to
clear the serialized handoff gate. Payload correctness PASS alone does not clear
the latency gate; a smaller `--rounds` run is a smoke test.

`doorbell` is a separate later pipeline gate. The worker queues a captured GPU
wait before the producer starts. The producer includes a `--delay-us 50` device
delay on every replay. Main polls the pinned GPU ring and checks the mapped
payload before either process queries or synchronizes the producer. Only then
may the worker synchronize and reuse its buffers. This tests unsynchronized
publication and a delayed producer, with an external deadline to kill hanging
device waits. Its reported latency includes the intentional delay, CPU
acknowledgments and mapped checks; do not compare it directly with the serialized
latency gate. `--delay-us 0` removes the delay for a separate measurement.

The launcher returns nonzero on an error or deadline and kills/reaps both
children. Logs include selected devices, host, mode, sizes and each child's last
64 KiB. Standalone child usage is `--role main|worker --fd N`; region size can be
queried with `--region-bytes` before any GPU initialization. Inputs are bounded;
unknown options and invalid values fail.

## Registration capacity gate

Do not run the full capacity test on a 31 GiB host. The default uses a shared
32 GiB memfd and an actually page-touched, private 16 GiB anonymous arena in
**each** process, simultaneously. This requires more than 64 GiB available RAM
plus OS/runtime overhead. Registration can also fail due to locked-memory limits.
The probe prints `RLIMIT_MEMLOCK` and `/dev/shm` capacity (memfd pages are not
allocated from the `/dev/shm` mount).

```sh
# Small smoke test first; cannot clear the 32 + 16 + 16 GiB capacity gate:
python3 tests/xvendor/run.py --cuda "$CUDA_EXE" --hip "$HIP_EXE" \
  --mode large-registration --registration-mib 64 --arena-mib 32 \
  --hip-libdir "$ROCM_LIB" --timeout 180
# Full gate, sufficiently large host only; repeat with --main hip:
python3 tests/xvendor/run.py --cuda "$CUDA_EXE" --hip "$HIP_EXE" \
  --mode large-registration --main cuda --hip-libdir "$ROCM_LIB" --timeout 900
```

A barrier proves both registrations and committed arenas remain alive together.
Each runtime verifies H2D/D2H on a registered 64 MiB chunk (or less for smoke
sizes) and reports repeated synchronous H2D bandwidth. This samples transfer
bandwidth, not a scan of every byte in the 32 GiB range. Registration and the
arenas remain alive through a final barrier; normal teardown unregisters before
unmapping. Failure is nonzero; the launcher reports an OOM-killed process too.

## In-process coexistence and visibility

Use fresh executions for every initialization/teardown combination. This loader
opens both runtime-only libraries with `RTLD_NOW | RTLD_LOCAL`, registers the
same host buffer simultaneously, and verifies real H2D/D2H through both runtimes.
It unregisters, shuts down and dlcloses the first library while the other remains
registered, then verifies the surviving runtime again before its teardown.

```sh
CUDA_LIB=/tmp/xvendor-cuda/tests/xvendor/libxprobe_cuda.so
HIP_LIB=/opt/strata-xvendor/hip/libxprobe_hip.so
LOADER=/tmp/xvendor-cpu/strata-xvendor-inproc
for order in cuda-first hip-first; do
  for teardown in cuda-first hip-first; do
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
      LD_LIBRARY_PATH="$ROCM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$LOADER" --cuda "$CUDA_LIB" --hip "$HIP_LIB" --order "$order" --teardown "$teardown"
  done
done
# Visibility diagnostics: CUDA_VISIBLE_DEVICES may also filter HIP. Repeat fresh
# for both orders; inspect reported vendor, count, device name and architecture.
for order in cuda-first hip-first; do
  env -u ROCR_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH="$ROCM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LOADER" --cuda "$CUDA_LIB" --hip "$HIP_LIB" --order "$order" --teardown hip-first
done
```

Each probe uses device zero **after** visibility filtering. A visibility failure
is diagnostic evidence, not a reason to claim both runtimes can share those
environment settings. The two-process launcher avoids that coupling. Successful
`dlclose` here is a spike test; it does not establish that an eventual engine
library should be unloaded.

## Report the gate, including missing checks

Record commit, same-day normal single-vendor baseline, OS/kernel, CPU/RAM,
GPU names/architectures/VRAM, drivers, CUDA/ROCm/compiler versions, image digest,
PCIe slots/link widths/NUMA topology, runtime paths and memlock limits. Include
commands, vendor direction, mode, round count, p50/p99, stale count, registration
and arena sizes, H2D bandwidth, all four loader combinations and visibility logs.
Use [COMMUNITY_BENCHMARKS.md](../../docs/COMMUNITY_BENCHMARKS.md) for reports and
search existing issues/PRs before opening one. State every toolchain/card/mode
that was not tested. If a gate fails or remains unmeasured, stop before engine
integration and report that gate explicitly.
