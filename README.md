# Simple Radeon Subs

Generate, clean, and translate movie subtitles on AMD Radeon GPUs under WSL.

The pipeline is:

```text
video -> FFmpeg audio extraction -> ASR -> subtitle cleanup -> translation
```

Choose between whisper.cpp (the default, built with HIP) and Qwen3-ASR with
Silero VAD and Qwen3-ForcedAligner. Both produce source-language SRT subtitles
for the same cleanup and local translation stages.

## Requirements

- x86-64 Linux or WSL2
- [uv](https://docs.astral.sh/uv/)
- system `ffmpeg` and `ffprobe` on `PATH`
- system ROCm with `hipcc` and `rocminfo`
- Git and CMake for the managed whisper.cpp build

The target environment is WSL2 with an AMD Radeon RX 9070 XT (`gfx1201`), system
ROCm 7.2.4, and the vLLM 0.27.1 `rocm723` wheel stack. Other compatible AMD
GPUs may work but are not part of the current tested baseline.

Install FFmpeg on Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
```

## Environment Setup

Clone the repository and install the project Python version:

```bash
git clone https://github.com/yiz-liu/Simple-Radeon-Subs.git
cd Simple-Radeon-Subs
uv python install 3.12
```

The Tsinghua mirror is the default package index, while vLLM is explicitly
sourced from its ROCm wheel index. `uv.lock` records the package versions and
sources used by the installation commands below.

Create the base environment from the committed lock file:

```bash
uv sync --locked --managed-python
```

This creates `.venv` with uv-managed Python 3.12. Activate it before running
repository commands:

```bash
source .venv/bin/activate
```

Verify the base environment and system tools:

```bash
python --version
python -c "import tqdm; print('base environment OK')"
ffmpeg -version
ffprobe -version
```

Prepare the full ROCm environment:

```bash
uv sync --locked --managed-python --group vllm
source .venv/bin/activate
.venv/bin/python scripts/patch_vllm.py
```

Prepare the default Whisper backend:

```bash
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

Or install the Qwen dependency group and prepare its weights:

```bash
uv sync --locked --managed-python --group qwen
source .venv/bin/activate
./scripts/download_weights.sh --asr-backend qwen
./scripts/doctor.sh --asr-backend qwen
```

Both download commands include the shared translation model. For Qwen ASR and
alignment, the script runs `uvx hf download` to fetch each complete repository at
a pinned revision into a staging directory, then verifies and installs the
required runtime files. The Hugging Face CLI runs in
an isolated uv tool environment; see the [official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli#using-uv).
Silero ONNX comes from its official release. Versions, file lists and SHA-256 hashes are defined in
`scripts/download_weights.sh`.
Existing files with unexpected checksums are reported without overwriting them.
To verify Qwen weights without downloading or using the GPU:

```bash
./scripts/download_weights.sh --asr-backend qwen --check
```

Run the minimal vLLM translation smoke test with the managed model profile:

```bash
python scripts/basic_translation.py
```

`uv sync` installs the locked Python dependencies. The scripts verify the
official vLLM wheel for WSL, build the pinned whisper.cpp CLI for the AMD
targets reported by `rocminfo`, and download verified weights. The doctor checks
the system tools, Python GPU stack and selected ASR backend. The download
script verifies ASR assets and the shared translation model.

`scripts/basic_translation.py` sends one batch of four translation requests.
Each request contains three French subtitle lines from a public-domain classic
and returns structured Chinese translations.

## Upgrading vLLM

The `vllm` group installs vLLM for translation; `qwen` includes that group and
adds ASR, VAD and alignment dependencies. For a Qwen environment, use
`--group qwen` in place of `--group vllm` in the upgrade commands below.

Choose a release with an official ROCm wheel compatible with the system ROCm,
Python, and GPU. Check the [vLLM releases](https://github.com/vllm-project/vllm/releases)
for the release's ROCm index URL.

Update these values together (the examples show the current version):

| File | Setting | Current value |
| --- | --- | --- |
| `pyproject.toml` | `vllm` entry under `[dependency-groups]` | `vllm==0.27.1` |
| `pyproject.toml` | URL of the `vllm-rocm` index | `https://wheels.vllm.ai/rocm/0.27.1/rocm723` |
| `scripts/patch_vllm.py` | `EXPECTED_VERSION` | `0.27.1+rocm723` |

Use the full installed wheel version for `EXPECTED_VERSION`, including the
ROCm suffix. From the repository root, regenerate the lock and install its
complete runtime and development dependencies:

```bash
source .venv/bin/activate
uv lock --managed-python
uv sync --locked --managed-python --group dev --group vllm
```

Verify the installed version, GPU runtime, translation, and project tests:

```bash
python -c "from importlib.metadata import version; print(version('vllm'))"
.venv/bin/python scripts/patch_vllm.py
./scripts/doctor.sh
python scripts/basic_translation.py
pytest
ruff check .
```

The demo loads the managed translation model and prints a generated response.
After validation, update the version in this README and record the package
versions and test results in `DEVLOG.md`. Commit the version declarations,
`uv.lock`, script, and documentation together. Generate `uv.lock` with uv rather
than editing package entries by hand.

To install an upgrade already recorded in the repository, run:

```bash
source .venv/bin/activate
uv sync --locked --managed-python --group dev --group vllm
.venv/bin/python scripts/patch_vllm.py
./scripts/doctor.sh
python scripts/basic_translation.py
```

## Audio Extraction

The base environment is sufficient to exercise FFmpeg audio extraction:

```bash
python -m src.audio /path/to/video.mkv -o output.wav
```

The output is 16 kHz, mono, 16-bit PCM WAV for downstream ASR processing.
Extraction preserves the source audio timeline by filling timestamp gaps with
silence. To regenerate an existing WAV, add `--force`:

```bash
python -m src.audio /path/to/video.mkv -o output.wav --force
```

## Full Pipeline

The full pipeline uses the selected transcription backend and local vLLM translation:

```bash
python run.py /path/to/video.mp4
python run.py /path/to/movie.mkv --output-dir ./subs --src-lang de --lang French
python run.py /path/to/video.mp4 --keep-temp
python run.py /path/to/video.mp4 --translated-only
python run.py /path/to/video.mp4 --disable-vad
python run.py /path/to/media-directory/
python run.py /path/to/media-directory/ --asr-backend qwen --keep-temp
```

Useful options:

- `--asr-backend`: `whisper` (default) or `qwen`
- `--src-lang`: source language code; auto-detected when omitted
- `--keep-temp`: preserve successful per-input sidecars containing intermediate WAV and SRT files
- `--force`: delete managed intermediates and rebuild every stage for that input
- `--translated-only`: omit source text from the final subtitles
- `--disable-vad`: transcribe the complete audio stream without integrated VAD

Directory inputs are scanned recursively for supported video and audio files,
including media in subdirectories. Managed sidecar directories are excluded;
symbolic links resolving to the same source are processed once. Pass a file path
to process only that file.

Directory runs are organized by stage: FFmpeg extracts every pending input,
the selected backend transcribes all pending audio files, cleanup runs for every
transcription, and vLLM translates all prepared requests with one model load.
Whisper loads once per batch; Qwen loads ASR once, then the aligner once when
needed. Qwen GPU workers exit before translation starts. Audio files and video
files can be mixed in the same input tree.
Within each Qwen stage, files are processed sequentially with the loaded model.
Each file submits all its ASR windows together; alignment submits all windows
that require timestamps together. vLLM schedules up to four concurrent requests
with an 8192-token budget per scheduling step.
When `--output-dir` is used, the source directory layout is preserved below it.

Each source uses a private sidecar beside the media file:

```text
.<source-filename>.simple-radeon-subs/
```

Intermediate SRT caches are separate for each backend, source language and VAD
mode; the extracted audio is shared. Existing, up-to-date intermediate stages
are reused after a failure. Successful jobs remove their sidecars unless
`--keep-temp` is set; failed jobs retain them.
An existing final SRT always skips the input before FFmpeg or either model is
initialized, so a successful input remains skippable after its sidecar has been
removed. Use `--force` when audio extraction settings, model settings, or weights
have changed, or when switching backends for an input that already has a final
SRT. Final output filenames are shared between backends. The process exits
nonzero if any input fails while still allowing other inputs in the same batch
to finish.

Translation is fully local and uses the managed model at
`models/Qwen3.5-9B-AWQ-4bit`. The application does not send subtitle content to
an online service.

## Local Translation

Translate a prepared SRT file independently with:

```bash
python -m src.translate /path/to/subtitles.srt -o translated.srt --lang Chinese
python -m src.translate /path/to/subtitles.srt --lang French --translated-only
```

The backend uses a fixed vLLM profile. Each request owns 16 subtitle cues and
includes up to four translated cues of context on either side. Structured
output preserves the cue mapping, and incomplete responses are rejected instead
of being written as partial subtitles. Generation is capped at 2048 tokens per
request; failures report the owned and contextual cue ranges. Request sizing is
intentionally not a CLI option.

## Standalone Transcription

Transcribe a prepared audio file with automatic source-language detection or an
explicit language code:

```bash
python -m src.transcribe /path/to/audio.wav -o ./subs
python -m src.transcribe /path/to/audio.wav -o ./subs -l de
python -m src.transcribe /path/to/audio.wav -o ./subs --quiet
python -m src.transcribe /path/to/audio.wav -o ./subs --disable-vad
python -m src.transcribe /path/to/audio.wav -o ./subs --asr-backend qwen -l fr
```

`--quiet` hides transcription progress; warnings and errors remain visible.
Qwen requires 16 kHz mono PCM16 WAV;
use `python -m src.audio` to prepare other formats. The command writes the SRT
to the selected output directory. Stage data is exchanged in a temporary
directory that is cleaned up after transcription, including on failure.

Qwen shows a file progress bar for each stage and window progress during ASR
and alignment. The inference bar counts completed audio windows, not subtitle
lines or audio seconds. Alignment
excludes short windows that use their existing boundaries. Model loading
has a separate status message. Translation shows completed requests.

Qwen ASR, alignment and translation show project stage messages and inference
progress. Project warnings and errors remain visible. Native runtime output is
buffered temporarily; failures include the last 8 KiB of available diagnostics.
Each failed Qwen file is reported once in the final summary, with a traceback
for unexpected exceptions. Worker failures include the exit code. Runtime output
is not retained as log files. Logs and progress use standard error; when it is
redirected or piped, progress uses ordinary lines instead of dynamic bars.

## Managed Qwen ASR Profile

The fixed assets are `models/Qwen3-ASR-1.7B`,
`models/Qwen3-ForcedAligner-0.6B`, and `models/silero-vad/silero_vad.onnx`.
The managed subtitle route supports Cantonese (`yue`), Chinese (`zh`), English
(`en`), French (`fr`), German (`de`), Italian (`it`), Japanese (`ja`), Portuguese
(`pt`), Russian (`ru`), and Spanish (`es`). Full language names also work.
Omitting the source language enables detection per window. Unsupported detected
languages fail the file with an error. This list is narrower than ASR-only
model support because subtitle timing also requires tokenization and alignment;
Korean tokenization requires a package outside the managed runtime.

The execution paths share input/output handling and separate their runtimes:

```text
Transcriber
  whisper -> WhisperBatchRunner -> whisper-cli
  qwen    -> QwenBatchRunner (parent)
               prepare -> QwenWorker -> WindowPreparer (CPU)
               asr     -> QwenWorker -> QwenASR
               align   -> QwenWorker -> ForcedAligner
               publish -> SRT
```

Each Qwen stage exits before the next starts. Within a stage, one worker reuses
its model across files. Source-language parsing belongs to QwenASR; process
launching and publication belong to QwenBatchRunner. `src/utils.py` contains the
atomic SRT writer shared by transcription and translation.

Fixed parameters live in `src/config.py`. Audio windows are built in
`src/audio.py`, transcription is managed by `src/transcribe.py`, and alignment
and timing rules live in `src/aligner.py`:

1. Silero ONNX runs on CPU: threshold `0.01`, minimum speech `250 ms`, minimum
   silence `100 ms`, padding `50 ms`, maximum speech `180 s`.
2. Neighboring speech spans separated by at most `1 s` share continuous audio
   context. Long spans are divided near low-energy points into roughly `30 s`
   windows, preserving sample offsets. `--disable-vad` processes the entire
   timeline through the same bounded windows.
3. Windows up to `3 s` become one cue without alignment. Longer windows use
   ForcedAligner; sentence-ending punctuation and semicolons determine subtitle
   boundaries. Decimal points stay within numbers. Character count only affects
   line wrapping, not subtitle timestamps.
4. Valid alignment is retained and bounded to the window. Consecutive invalid
   spans use neighboring valid cues or the window edges. Gaps below `0.32 s`
   merge into an adjacent cue in the same window, preserving text order.
5. Each file's windows are transcribed in one pass. Truncated ASR or extreme
   consecutive repetition fails the file during ASR and reports the affected
   time ranges. Native errors also fail the file.
   Empty ASR windows produce a warning. Timing fallback is an estimate and still
   benefits from listening checks. Qwen cleanup preserves long cue durations.

ASR and alignment use BF16, eager execution, model-runner V1, four concurrent
requests, an `8192`-token context/batch budget and `0.5` GPU memory utilization.
ASR uses greedy decoding with a `4096`-token output limit and
`repetition_penalty=1.2`. Repeat detection uses the pinned Qwen processor's
extreme-repeat rules to identify invalid output.

## Subtitle Cleanup

Run conservative SRT cleanup independently with:

```bash
python -m src.clean /path/to/subtitles.srt -o cleaned.srt
python -m src.clean /path/to/subtitles.srt -o cleaned.srt --disable-vad
```

Cleanup replaces malformed UTF-8 with `U+FFFD`, trims surrounding whitespace,
and removes empty or punctuation-only cues. It removes four or more identical
effective characters, and removes text of at least 32 effective characters when
a 2-8 character phrase repeats consecutively at least four times. Remaining
exact repetitions of a 1-8 character unit are limited to three consecutive
copies. Two identical consecutive cues are merged when their gap is at most 250
ms; close runs of three or more identical cues are discarded. For Whisper
VAD-derived subtitles, every cue longer than eight seconds keeps its end timestamp and moves
its start timestamp so that only its final eight seconds remain. `--disable-vad`
preserves long cue timestamps. Other content is preserved.

## Managed whisper.cpp ASR Profile

Each backend uses a fixed profile with managed models and devices. Whisper
assets are:

```text
.venv/bin/whisper-cli
models/whisper/ggml-large-v3-turbo.bin
models/whisper/ggml-silero-v6.2.0.bin
```

The default quality-oriented decoder settings are:

```text
model: large-v3-turbo
processors: 1
beam size: 5
max context: 48
non-speech token suppression: disabled
VAD: enabled
```

The default integrated-VAD settings use the best profile from the three-sample
comparison:

```text
max context: 48
VAD threshold: 0.05
VAD minimum speech: 100 ms
VAD minimum silence: 120 ms
VAD maximum speech: 10 s
VAD speech padding: 500 ms
VAD sample overlap: 0 s
```

`--disable-vad` changes max context to `0`, enables non-speech token
suppression, and transcribes the complete audio stream without the VAD model.

These settings are internal and fixed. Input files in a directory batch are
processed one after another by `whisper-cli` while reusing the loaded model.

## Known Issues

### Long-form recognition

Audio extraction preserves source timestamps, including gaps. Filling gaps can
change Whisper decoding behavior; a long-form comparison showed substantial
repetition with the current Whisper VAD profile. Review long-form output and
compare the Qwen backend when this occurs. Rebuild older sidecars with `--force`
to apply the current extraction settings.

### whisper.cpp VAD timestamps

When built-in VAD removes a long silence, whisper.cpp may decode across
the compressed boundary and map one short subtitle over the entire original gap.
This is tracked upstream in
[#3584](https://github.com/ggml-org/whisper.cpp/issues/3584) and
[#3634](https://github.com/ggml-org/whisper.cpp/issues/3634).

As a local workaround, subtitle cleanup shortens every Whisper VAD-derived cue
longer than eight seconds to its final eight seconds while preserving the original end
timestamp. Use `--disable-vad` to preserve long cue timestamps.

### whisper.cpp invalid UTF-8

large-v3-turbo can occasionally end a segment on an incomplete byte-level token
and write invalid UTF-8. This remains open upstream in
[#3760](https://github.com/ggml-org/whisper.cpp/issues/3760).

The cleanup boundary decodes malformed input with replacement, logs the byte
position, and continues with conservative filtering.

## Development

Development tools are declared in the `dev` group. Install both transcription
backends and the test tools with:

```bash
uv sync --locked --managed-python --group dev --group qwen
source .venv/bin/activate
```

Run the project checks from the activated environment:

```bash
pytest
ruff check .
basedpyright --level error run.py src tests
```

`basedpyright` is an optional external developer tool and is not installed into
the project environment. Install it through the editor toolchain or npm when
type checking is required; Node.js is not a runtime requirement for the
application.

Project dependencies must be declared in `pyproject.toml`. Do not install
undeclared packages directly into `.venv`. Commit `uv.lock` whenever dependency
resolution changes.

## Troubleshooting

If audio extraction reports that FFmpeg is missing, confirm that both binaries
are discoverable:

```bash
command -v ffmpeg
command -v ffprobe
```

ROCm and the AMD driver are system resources and are not managed by uv. The ASR
and translation weights are repository-managed files under `models` and are
prepared by `scripts/download_weights.sh`.
