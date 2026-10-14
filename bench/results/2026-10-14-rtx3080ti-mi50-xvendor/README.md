# RTX 3080 Ti + Vega 20: cross-vendor Phase 0 tests

Measured on 2026-10-14. Mixed-vendor payload checks, GPU doorbells, small
registration tests and in-process runtime coexistence passed. The serialized
handoff **fails the plan's p99 < 200 us gate**. The full 32 + 16 + 16 GiB
registration gate was not run: this host has 30.51 GiB total physical memory
as reported by `MemTotal`, and a 3.81 GiB per-process memlock limit.

These Phase 0 measurements did not test model inference, MTP, prompt caches,
or API behavior. The original latency and registration-capacity gates remain
failed and unmeasured, respectively. After the user accepted the measured
latency, a separate experimental CLI implementation passed IQ3_S recall on a
149,900-token prompt with 150,000 Q4 context cells; see the
[model report](model/README.md) for that later test and its restricted scope.

## Machine and builds

| Item | Configuration |
|---|---|
| CPU | AMD Ryzen 5 7600X, 6 cores |
| Host memory | `MemTotal=31988996 kB` (30.51 GiB); 8 GiB swap |
| NVIDIA | RTX 3080 Ti, 12 GiB, sm_86; driver 595.84; power limit 350 W |
| AMD | Vega 20 / MI50-class, gfx906:sramecc+:xnack-, PCI 1002:66a1; reported VRAM 31.98 GiB; power limit 225 W |
| OS | Ubuntu 24.04, kernel 7.0.0-34-generic, glibc 2.39 |
| CUDA | 12.0.140; CUDA host compiler GCC 12.4 |
| HIP | Native ROCm 6.3.3, HIP 6.3.42134, clang 18 |
| Source | `40bcc2ca4f971902559f3794273ca2946748355e` plus [scalar-cache.patch](logs/scalar-cache.patch) |
| Source hash | `262e52b2d272b3d3d07acb461c8ccd2c9f6f0b617e408246b9402297fa6c4054` for `tests/xvendor/xvendor_handoff.cu` |

[hardware.json](logs/hardware.json) contains the captured hardware and versions.
The NVIDIA endpoint supports PCIe 4.0 x16; it downshifted to PCIe 1.0 x16 when
idle. The AMD endpoint reports PCIe 4.0 x16, but its upstream path includes
several PCIe 4.0 **x4** bridges. Endpoint width alone does not describe this path.
No clock, power, driver, or PCIe settings were changed. GPU clocks were automatic;
background operating-system activity was not controlled.

CUDA and gfx906 HIP probe builds succeeded using the installed toolchains.
The separate RDNA HIP gfx1100 variant also compiled and linked with ROCm 6.3.3;
no RDNA GPU execution was performed. CUDA used
`STRATA_WERROR=ON`. HIP initially failed strict compilation on existing unchecked
HIP return values in `src/core/pinned.cu`; the normal `STRATA_WERROR=OFF` build
succeeded. Probe destructors also emit unused-result warnings on HIP. Both probe
libraries export only `strata_xprobe_v1`.

The community ROCm 7.14 container was not used: the available Docker Desktop
daemon was stopped and the native Docker socket required additional permissions.
These results establish the tested native 6.3.3 configuration only.

## Serialized handoff after the counter fix

Each row has three fresh process-pair runs, 10,000 rounds per run. All values
below are microseconds. Medians summarize per-run p50/p99 values; ranges show all
three per-run p99 values. Every input, return value and index was checked.

| Main → worker | Wait | Median p50 | Median p99 | p99 range | < 200 us gate |
|---|---|---:|---:|---:|---|
| CUDA → HIP | spin | 269.000 | 281.090 | 280.950–285.250 | Fail |
| CUDA → HIP | futex | 283.260 | 299.400 | 298.780–301.740 | Fail |
| HIP → CUDA | spin | 259.290 | 263.959 | 263.850–272.131 | Fail |
| HIP → CUDA | futex | 262.520 | 283.660 | 280.840–288.900 | Fail |

The shared region is 749,568 bytes: 8 x 12,804 FP32 values outbound, 8 x 10,240
FP32 values and eight int32 indices back, with page-aligned areas. Graphs replay
with changing round values. Timing excludes the main process's host fill/upload
and final readback/check; it includes both GPU graph directions, CPU rendezvous
and the worker's processing/check. It is not a measurement of inference latency.

The observed x4 upstream path and the measured HIP H2D rate below may affect
these timings; their individual contribution was not isolated. No card-order
default or acceptance threshold was changed.

## Doorbell correctness

Three fresh runs of 10,000 rounds in each direction passed unsynchronized mapped
payload observation and the delayed-producer GPU wait. These are a separate
pipeline correctness check; their timing includes the configured device delay,
CPU acknowledgments and mapped payload checks.

| Main → worker | Median p50 us | Median p99 us | p99 range us |
|---|---:|---:|---:|
| CUDA → HIP | 1195.829 | 1209.800 | 1206.540–1237.741 |
| HIP → CUDA | 1154.021 | 1176.821 | 1167.679–1177.707 |

The delay parameter was 50 us. It uses nominal GPU clock rate; actual delay
duration was not calibrated. These doorbell timings do not use the serialized
200 us acceptance threshold. Across serialized and doorbell configurations,
**180,000 mixed-vendor rounds passed with zero stale values** after the fix.

## Measured probe bug and fix

The original synthetic validator read the same mapped host round counter in
every GPU thread. On the RTX, the identical validator took approximately
1788 us with that pointer versus 5 us with a device-resident counter. The
[standalone diagnostic](scalar_diagnostic.cu) alternated both paths twice in one
process, with five warm-ups and 50 measured rounds per path; it checked changing
input, all returns, indices and the error result. CUDA events timed the kernel.
See [cuda-scalar-diagnostic.log](logs/cuda-scalar-diagnostic.log).

The worker graph now copies those four bytes into device memory once before
validation. Shapes, payload checks, graph replay, timing extent and gate criteria
remain unchanged. Both GPU backends were rebuilt before the reported reruns.

Original uncached mixed runs are retained: CUDA → HIP p99 was 388.509/401.569 us
for spin/futex, and HIP → CUDA was 2424.614/2421.944 us. These original timings
include the validator artifact and should not be used as corrected transport
measurements. Original CUDA-only control timings included the same artifact.

## Same-day controls

These controls use the corrected build with two processes on **one physical
GPU**, one 10,000-round run each. They do not represent a split across two
same-vendor cards, and no model baseline was available.

| Physical GPU | Wait | p50 us | p99 us | Payload check |
|---|---|---:|---:|---|
| RTX 3080 Ti, both processes | spin | 259.180 | 274.510 | Pass |
| RTX 3080 Ti, both processes | futex | 268.489 | 272.770 | Pass |
| Vega 20, both processes | spin | 405.349 | 415.419 | Pass |
| Vega 20, both processes | futex | 419.509 | 434.029 | Pass |

## Registration and runtime coexistence

Both vendor orders passed with the same 64 MiB shared region and 32 MiB committed
private arena in each process. Both also passed with the same **2 GiB** registered
region and **512 MiB private arena each**, held alive together through barriers.
In the latter tests, a registered 64 MiB chunk was copied H2D 16 times and checked
with D2H readback: NVIDIA measured 26.648–26.694 GB/s; HIP 3.588–3.598 GB/s.
Those are synchronous wall-time measurements, one run per order, not a scan of
the entire 2 GiB region. The full 64 GiB physical-memory test was not attempted.

All four fresh in-process initialization/teardown combinations passed simultaneous
registration, H2D/D2H and a surviving-runtime transfer after the other library's
unregister/reset/dlclose. This tests the narrow probe ABI, not unloading an engine
library with its own threads and resources.

Visibility tests with both vendors selecting device 0 passed in both init orders.
A negative control with `CUDA_VISIBLE_DEVICES=1` and no `HIP_VISIBLE_DEVICES`
selected the AMD **gfx1036 iGPU** when HIP initialized first, while CUDA then
reported no device. The runner's removal of CUDA visibility from the HIP child's
environment is necessary on this machine. The iGPU was not used for handoff or
inference tests.

## Reproduce

Use the recorded base commit plus the included patch. From the repository root:

```sh
cmake -S . -B build-xvendor-cuda -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DSTRATA_XVENDOR_TESTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_WERROR=ON
cmake --build build-xvendor-cuda --target strata-xvendor-handoff xprobe_cuda -j 3
cmake -S . -B build-xvendor-hip -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DSTRATA_XVENDOR_TESTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_WERROR=OFF
cmake --build build-xvendor-hip --target strata-xvendor-handoff xprobe_hip -j 2
python3 tests/xvendor/run.py \
  --cuda build-xvendor-cuda/tests/xvendor/strata-xvendor-handoff \
  --hip build-xvendor-hip/tests/xvendor/strata-xvendor-handoff \
  --main cuda --mode spin --rounds 10000 --hip-libdir /opt/rocm/lib --timeout 180
```

Repeat `--main hip` and `--mode futex`/`doorbell` in fresh runs. The
[probe guide](../../../tests/xvendor/README.md) gives loader and registration
commands. [results.json](results.json) lists all 18 corrected mixed runs and the
captured machine facts; [logs/](logs/) retains builds, controls, original runs,
reruns and failure diagnostics. Home/repository paths were replaced with
`<TARGET_HOME>`/`<TARGET_REPO>` in logs.

Not tested: full registration capacity, ROCm 7.14/container deployment, RDNA
hardware, SYCL builds, inference or token parity, other PCIe layouts, multiple physical GPUs of
one vendor, manual clock/affinity tuning, or worker-crash tests inside an engine.
The existing CPU transport, bounded CLI and five launcher tests also passed on
this host. Latency remains failed and full capacity remains unmeasured: do not
advance to engine integration on the strength of these correctness checks.
