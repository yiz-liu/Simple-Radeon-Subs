# Simple Radeon Subs

Generate, clean, and translate movie subtitles on AMD Radeon GPUs under WSL.

The pipeline is:

```text
video -> FFmpeg audio extraction -> ASR -> subtitle cleanup -> translation
```

The ASR backend is whisper.cpp built with HIP. Its built-in Silero VAD uses
a fixed profile by default.

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

Prepare the full ROCm environment and the managed ASR runtime:

```bash
uv sync --locked --managed-python --group vllm
source .venv/bin/activate
.venv/bin/python scripts/patch_vllm.py
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

Run the minimal vLLM translation smoke test with the managed model profile:

```bash
python scripts/basic_translation.py
```

`uv sync` installs the locked Python dependencies. The scripts verify the
official vLLM wheel for WSL, build the pinned whisper.cpp CLI for the AMD
targets reported by `rocminfo`, and download verified weights. The doctor checks
the system tools, Python GPU stack, whisper-cli, and ASR assets; the download
script verifies the translation model.

`scripts/basic_translation.py` sends one batch of four translation requests.
Each request contains three French subtitle lines from a public-domain classic
and returns structured Chinese translations.

## Upgrading vLLM

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

## Full Pipeline

The full pipeline uses the managed whisper.cpp and local vLLM backends:

```bash
python run.py /path/to/video.mp4
python run.py /path/to/movie.mkv --output-dir ./subs --src-lang de --lang French
python run.py /path/to/video.mp4 --keep-temp
python run.py /path/to/video.mp4 --translated-only
python run.py /path/to/video.mp4 --disable-vad
python run.py /path/to/media-directory/
```

Useful options:

- `--src-lang`: source language code; whisper.cpp auto-detects it when omitted
- `--keep-temp`: preserve successful per-input sidecars containing intermediate WAV and SRT files
- `--force`: delete managed intermediates and rebuild every stage for that input
- `--translated-only`: omit source text from the final subtitles
- `--disable-vad`: transcribe the complete audio stream without integrated VAD

Directory runs are organized by stage: FFmpeg extracts every pending input,
whisper.cpp transcribes all pending audio files with one model load, cleanup runs
for every transcription, and vLLM translates all prepared requests with one
model load. Audio files and video files can be mixed in the same input tree.
When `--output-dir` is used, the source directory layout is preserved below it.

Each source uses a private sidecar beside the media file:

```text
.<source-filename>.simple-radeon-subs/
```

Existing, up-to-date intermediate stages are reused after a failure. Successful
jobs remove their sidecars unless `--keep-temp` is set; failed jobs retain them.
An existing final SRT always skips the input before FFmpeg or either model is
initialized, so a successful input remains skippable after its sidecar has been
removed. Use `--force` when fixed model settings or weights have changed and the
result must be rebuilt. The process exits nonzero if any input fails while still
allowing other inputs in the same batch to finish.

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
```

The command displays native transcription progress by default. `--quiet` hides
the Python progress bar without changing transcription behavior. Full-pipeline
directory runs submit every pending audio file to one whisper.cpp process.

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
ms; close runs of three or more identical cues are discarded. For VAD-derived
subtitles, every cue longer than eight seconds keeps its end timestamp and moves
its start timestamp so that only its final eight seconds remain. `--disable-vad`
preserves long cue timestamps. Other content is preserved.

## Managed whisper.cpp ASR Profile

The project intentionally exposes one fixed ASR profile rather than public
model or device selection. Its managed assets are:

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

### whisper.cpp VAD timestamps

When built-in VAD removes a long silence, whisper.cpp may decode across
the compressed boundary and map one short subtitle over the entire original gap.
This is tracked upstream in
[#3584](https://github.com/ggml-org/whisper.cpp/issues/3584) and
[#3634](https://github.com/ggml-org/whisper.cpp/issues/3634).

As a local workaround, subtitle cleanup shortens every VAD-derived cue longer
than eight seconds to its final eight seconds while preserving the original end
timestamp. Use `--disable-vad` to preserve long cue timestamps.

### whisper.cpp invalid UTF-8

large-v3-turbo can occasionally end a segment on an incomplete byte-level token
and write invalid UTF-8. This remains open upstream in
[#3760](https://github.com/ggml-org/whisper.cpp/issues/3760).

The cleanup boundary decodes malformed input with replacement, logs the byte
position, and continues with conservative filtering.

## Development

Development tools are declared in the `dev` group and are not installed by the
default sync. Create a development environment with:

```bash
uv sync --locked --managed-python --group dev --group vllm
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
