# Development Log: ROCm Environment

This log preserves the environment experiments that led to the current WSL and
AMD GPU setup.

## Initial Experiments

The initial target was local movie subtitle generation and translation on:

- WSL2 with Ubuntu;
- AMD Radeon RX 9070 XT (`gfx1201`);
- ROCm 7.2.0.

### CTranslate2 ROCm fork

The first ASR route used a ROCm fork of
[CTranslate2](https://github.com/OpenNMT/CTranslate2), the runtime behind
`faster-whisper`.

The experiment used:

- repository: `https://github.com/arlo-phoenix/CTranslate2-rocm.git`;
- compiler: AMD clang++ from ROCm 7.2.0;
- target: `gfx1201`.

Configuration required CMake policy adjustments and replacing an unavailable
Intel OpenMP dependency (`libiomp5`) with `libomp`. Compilation then failed in
the CUDA-to-HIP compatibility layer, including:

```text
error: no template named 'counting_iterator' in namespace 'thrust'
error: use of undeclared identifier 'hipblasGemmEx_v2'
```

The available fork depended on Thrust and hipBLAS APIs that did not match the
installed ROCm headers. Adapting and maintaining that native fork was outside
the project scope, so this route was abandoned. This result applies to the
tested fork and toolchain; it is not a general claim that CTranslate2 cannot run
on ROCm.

### Native PyTorch and openai-whisper

The next experiment installed ROCm-specific PyTorch wheels and ran the official
`openai-whisper` package with `large-v3-turbo`. It successfully used the RX 9070
XT and established that the WSL driver path, PyTorch ROCm runtime, and Whisper
weights were functional.

This became the first working ASR baseline. It was later superseded as the
runtime boundary because long-form decoding remained inside Python and did not
provide the desired native execution profile.

### Early toolchain design

The first working design proposed:

- a downloaded static FFmpeg under `tools/ffmpeg/`, intended to avoid `sudo`
  and differences between system packages;
- `openai-whisper` with `large-v3-turbo`, producing SRT directly;
- regex-based cleanup for hallucinated credits such as “Subtitle by Amara.org”
  and SDH text such as `[Music]`;
- local Qwen translation through vLLM, with requests submitted in batches to
  improve GPU use and keep subtitle content on the workstation.

These decisions captured a functional prototype, not a reproducible environment
contract. In particular, a bundled FFmpeg introduced another platform-specific
binary and update mechanism, while Python dependencies, native builds, and model
weights still lacked clear ownership.

## Current Environment Baseline (2026-07-19)

The environment work was revisited with the following completion criteria:

- a locked Python 3.12 environment;
- working PyTorch and vLLM ROCm extensions;
- a reproducible HIP build of whisper.cpp;
- verified model files at fixed project paths;
- repeatable checks for the assembled runtime and downloaded assets.

The verified machine now uses system ROCm 7.2.4. System ROCm did not need to be
downgraded to match the patch version encoded by the Python wheels.

### Dependency ownership

The final design separates three dependency layers:

| Layer | Owner | Contents |
| --- | --- | --- |
| System | OS and ROCm installation | FFmpeg, CMake, Git, `hipcc`, `rocminfo` |
| Python | uv and `uv.lock` | Python 3.12, ModelScope, test tools, vLLM and its ROCm stack |
| Native assets | Project scripts | whisper.cpp CLI, build manifest, model weights |

uv manages Python packages only. Project scripts build or download external
runtime assets, and the operating system provides drivers and general-purpose
native tools.

### Python package resolution

Building vLLM from source was considered when no ROCm 7.2.4 wheel was apparent.
The vLLM wheel index provides a coordinated `rocm723` stack for vLLM 0.27.1,
including PyTorch, Triton, AITER, and compiled ROCm extensions. Reusing that
stack was more reproducible than rebuilding the same dependency graph.

`pyproject.toml` uses the Tsinghua mirror as the default index and binds vLLM to:

```text
https://wheels.vllm.ai/rocm/0.27.1/rocm723
```

uv uses `first-index`, and the committed lock records both sources. A locked
sync therefore requires no local index override and cannot silently resolve
vLLM from a general-purpose index.

The verified packages are:

```text
vLLM: 0.27.1+rocm723
PyTorch: 2.11.0+gitd0c8b1f
Triton: 3.6.0
Torch HIP runtime: 7.2.53211
```

The system ROCm 7.2.4 compiler remains responsible for native HIP builds. Its
patch version and the Python wheel stack do not need to be identical when the
runtime compatibility checks pass.

### vLLM on WSL

The prebuilt wheel loads and computes on the RX 9070 XT, but WSL blocks the AMD
SMI path used by parts of vLLM platform discovery. Two failures were observed:

1. ROCm detection could stop before falling back to `torch.version.hip`.
2. Early `warning_once` calls could import distributed state while platform
   initialization was incomplete, causing a circular import.

The vLLM check and WSL fallback are now implemented by
`scripts/patch_vllm.py`; it verifies the installed `0.27.1+rocm723` wheel and
runs the GPU import smoke test before applying any modification.

The WSL fallback is provided upstream by
[#38434](https://github.com/vllm-project/vllm/pull/38434).

### Managed whisper.cpp build

whisper.cpp is neither a Python dependency nor a retained submodule.
`scripts/install_whisper_cli.sh` performs a transient, pinned build:

1. fetch the pinned commit into a temporary directory;
2. detect AMD targets through `rocminfo`;
3. build only `whisper-cli` with HIP in Release mode;
4. reject unexpected shared whisper.cpp dependencies;
5. install the executable at `.venv/bin/whisper-cli`;
6. record commit, target, toolchain, and binary hashes in
   `.venv/.whisper-cli-build.json`;
7. remove the temporary source and build trees.

This provides a reproducible native ASR tool without keeping a second source
tree in the repository or treating CMake output as a uv package.

### System FFmpeg

The project-local static FFmpeg proposal was retired. `ffmpeg` and `ffprobe` are
system prerequisites and must be present on `PATH`.

This avoids storing a large third-party binary and avoids adding a separate
platform-specific download mechanism. The application verifies both tools and
uses FFmpeg's machine-readable progress output.

### Model storage

Large weights are managed outside uv under fixed project paths:

```text
models/whisper/ggml-large-v3-turbo.bin
models/whisper/ggml-silero-v6.2.0.bin
models/Qwen3.5-9B-AWQ-4bit/
```

ModelScope is a base Python dependency and provides the primary download path
for the Whisper and translation models. The Silero VAD file comes from the
official whisper.cpp Hugging Face repository. `scripts/download_weights.sh`
pins the translation revision, checks required files and SHA-256 hashes, and
refuses to overwrite unexpected existing content. Checksums also fix the exact
Whisper and VAD artifacts when their source URLs use moving branch names.

Weights are prepared after the software environment. They are runtime assets,
not Python packages, and do not belong in `uv.lock`.

### Reproduction and verification

The environment can be reproduced with:

```bash
uv python install 3.12
uv sync --locked --managed-python --group dev --group vllm
source .venv/bin/activate
.venv/bin/python scripts/patch_vllm.py
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

The doctor checks system tools, Python 3.12, ModelScope, PyTorch, Triton, vLLM,
ROCm platform selection, BF16 GPU computation, native vLLM extensions,
whisper-cli linkage, and the two ASR weights. The download script separately
verifies the translation model.

At this point each environment component had one owner and a reproducible
verification path. This completed the environment-preparation phase.

## 2026-09-22: v0.27.1 WSL ROCm patch check

The managed vLLM profile is now pinned to `0.27.1+rocm723` with Torch 2.11.
`scripts/patch_vllm.py` checks the installed runtime first and exits without
modifying vLLM when it already selects `RocmPlatform`. If AMD SMI detection
fails under WSL, it applies the v0.27.1 fallback and rechecks the platform and
device name. `scripts/patch_vllm.py` is the check-first ROCm patch entry point.

The translation smoke test is now `scripts/basic_translation.py`. It submits
four requests in one vLLM batch, with three French subtitle lines from classic
works in each request, and prints the four structured translation responses.

## 2026-09-25: source timeline and Qwen transcription backend

FFmpeg extraction now uses `AUDIO_TIMELINE_FILTER` from `src.config`:
`aresample=16000:async=1:first_pts=0:min_hard_comp=0.001`. It fills or trims
samples to preserve source timestamps without soft tempo compensation.
Regression fixtures cover continuous audio and 25 ms / 500 ms timestamp gaps.
A long-form control showed that this change can trigger substantial repetition
with the fixed Whisper VAD profile, even though the extracted timeline is
correct. This remains a recognition limitation, documented in the README.

`--asr-backend qwen` routes the existing transcription interface through local
Silero ONNX, Qwen3-ASR and Qwen3-ForcedAligner. Whisper remains the default.
Qwen audio preparation lives in `src/audio.py`, batch orchestration and ASR in
`src/transcribe.py`, and forced alignment and timing in `src/aligner.py`. Batch
processing reuses one ASR model load and one aligner model load, in separate processes that exit before translation.
Backend-specific SRT sidecars share the canonical WAV. Qwen cleanup preserves
validated long cue boundaries instead of applying the Whisper VAD duration cap.

The promoted timing rules use short windows directly and repair invalid long
window alignment within neighboring boundaries. A preceding full-length
prototype comparison reduced aligner requests from 564 to 163 and total time
from 258.09 to 239.61 seconds. These are single-run prototype measurements,
not throughput claims for the integrated backend. Empty transcripts and recovered
cue origins remain explicit in diagnostics.

The weight downloader pins official model revisions and verifies runtime files
with SHA-256. `scripts/download_weights.sh --asr-backend qwen` prepares the full
model set; `scripts/doctor.sh --asr-backend qwen` selects the matching checks.
The `vllm` group declares vLLM itself. The `qwen` group includes `vllm` and
declares Silero, tokenization, array processing, result serialization and
processor dependencies. Transformers is pinned to the native Qwen processor
version used by this route. Weight lists and hashes are stored directly in the
Shell downloader, whose `--check` mode is also used by the doctor.
No package installation or runtime patch was performed during integration.

Integration GPU validation was attempted but stopped before inference because
7.30 GiB was free and the fixed profile required 7.91 GiB. Further GPU tests were
deferred to manual validation. CPU regression tests, static checks, CLI boundary
checks and pinned local weight verification cover the integration separately.

After consolidation, the submission test set passed 101 tests; two GPU
integration tests remain reserved for manual validation. The complete local
suite passed 152 non-GPU tests. Ruff, basedpyright, LSP diagnostics, shell syntax
and lock consistency checks passed. Existing pinned weights passed the unified
Shell check. Offline replay of 567 existing records reproduced all 833 subtitle
cues exactly, including text, timestamps and timing origins. A real child-process
check confirmed that short windows skip model loading, and importing the default
transcription interface does not import the Qwen runtime dependencies.

## 2026-09-25: transcription ownership and weight download entry

Qwen model downloads now use `uvx hf download` without filename arguments to fetch
complete repositories at pinned revisions into staging directories inside the
existing Shell downloader. Only verified runtime files are installed. Local
checksum checks and publication safeguards remain in the script; `--check` never
invokes uvx.
The shared translation model retains its existing download source.

`Transcriber` selects the backend, `QwenBatchRunner` launches stage processes,
`QwenWorker` owns each stage's model lifecycle, and `QwenASR` handles decoding and
language parsing. Path-specific stateless helpers now live on their owning
classes. `src/utils.py` owns atomic SRT saving for transcription and translation.
Configuration constants are grouped by stage without changing their values.
Named exception subclasses retain inherited behavior and document the failures
they identify. GPU inference remains deferred to manual validation.

The submission suite passed 103 tests with two GPU tests excluded; the complete
local suite passed 154 non-GPU tests. Static checks and existing-weight checks
passed. An offline replay retained the same 833 cues from 567 records, and the
short-window child-process check loaded no inference model. HF invocation was
verified through the Shell script using a controlled CLI substitute; no weights
needed downloading in the installed environment.

Directory job planning now deduplicates resolved source paths before extraction,
so symbolic-link aliases cannot create duplicate extraction tasks or inflate
batch counts. Recursive discovery is explicit in CLI help and the README.
Qwen diagnostic-directory announcements were initially moved to DEBUG. The
production cleanup below subsequently removed their creation entirely.

## 2026-09-25: remove Qwen experiment diagnostics from production

Removed persistent `raw.qwen/run-*` directories, stage reports, copied native
logs, `cues.json`, `run.json`, performance counters, raw timestamp decoding and
cue provenance metadata. Model reuse and subtitle timing rules remain unchanged.
The parent sends job inputs through stdin; stage results and per-file errors
use a scoped temporary directory that is removed on success, failure or a
handled interruption. Native failures report the exit code and a bounded output
tail instead of referring to retained log files. WAV and SRT pipeline caches
continue to follow the existing sidecar policy.

Validation passed 109 submission tests with two GPU tests excluded, plus Ruff,
basedpyright and LSP checks. Offline replay of 567 saved records reproduced all
833 cues exactly in text, timestamps and validity. A real short-window child
process completed without loading an inference model; cleanup tests cover
success, worker failure and interruption. GPU validation remains manual.

## 2026-09-25: retry truncated or repetitive ASR windows

A targeted three-window GPU investigation reproduced output-token exhaustion
through repeated text. Each window generated 4096 tokens while remaining below
the model context limit. Repetition penalty 1.1 stopped the observed loops;
changing the prompt alone still allowed a highly repetitive stop-ended result.

The first ASR pass retains its original decoding profile. Windows ending at the
length limit or flagged by the pinned Qwen processor's extreme-repeat logic
are retried once with repetition penalty 1.1 in the same model instance. Only
those windows are resubmitted, using the original audio and language settings.
The repetition helper is used for detection; its rewritten text is not
published. Unrecovered windows fail during ASR with IDs and time ranges, before
alignment starts for that file. No diagnostic artifacts or dependencies were
added. The integrated retry is reserved for manual GPU testing.

Validation passed 115 submission tests with two GPU tests excluded, plus Ruff,
basedpyright and LSP checks. Regression coverage verifies selective single
retries, preserved audio/language settings, ordinary repetition and early ASR
failure after an unsuccessful retry.

## 2026-09-26: show Qwen batch progress

Qwen stage workers now display file progress and use vLLM's native progress
callbacks for request preparation and completed audio windows. ASR retries
have a separate label; alignment counts only windows that require alignment.
A dedicated inherited terminal descriptor keeps progress visible while native
stdout/stderr remain captured for bounded failure reporting. `--quiet` disables
the progress channel. The callbacks do not change request batch sizes, model
reuse, concurrency limits, timing rules or the temporary-data cleanup policy.

Validation passed 126 submission tests with two GPU tests excluded, plus Ruff,
basedpyright and LSP checks. Real PTY subprocess checks observed partial progress
before worker exit, verified quiet mode and native-log separation, and covered
40-column, 120-column and zero-width terminal reports (falling back to 80).
Narrow terminals retain labels and counters while omitting timing metadata.
GPU inference remains reserved for manual validation.
