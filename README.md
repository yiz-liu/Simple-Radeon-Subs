# Simple Radeon Subs

Generate, clean, and translate movie subtitles on AMD Radeon GPUs under WSL.

The pipeline is:

```text
video -> FFmpeg audio extraction -> ASR -> subtitle cleanup -> translation
```

The integrated ASR backend is whisper.cpp built with HIP, using its built-in
Silero VAD. The standalone Python Silero stage has been removed.

## Requirements

- Linux or WSL2
- [uv](https://docs.astral.sh/uv/)
- System `ffmpeg` and `ffprobe` available on `PATH`
- ROCm-compatible AMD GPU for ASR acceleration
- Google Gemini API key when translation is enabled

Install FFmpeg on Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
```

The project does not download or bundle FFmpeg.

## Environment Setup

Clone the repository and install the project Python version:

```bash
git clone https://github.com/yiz-liu/Simple-Radeon-Subs.git
cd Simple-Radeon-Subs
uv python install 3.12
```

When creating the lock file for the first time, or after editing dependency
declarations directly, run:

```bash
uv lock --managed-python
```

Create the base environment from the committed lock file:

```bash
uv sync --locked --managed-python
```

This creates `.venv` with uv-managed Python 3.12. Activating it is optional:

```bash
source .venv/bin/activate
```

Verify the base environment and system tools:

```bash
uv run --managed-python python --version
uv run --managed-python python -c "import dotenv, tqdm; print('base environment OK')"
ffmpeg -version
ffprobe -version
```

Create the local configuration file when translation is needed:

```bash
cp .env.template .env
```

Add `GEMINI_API_KEY` to `.env`. The file is ignored by Git.

Prepare the full ROCm environment and the managed ASR runtime:

```bash
uv sync --locked --managed-python --group vllm
source .venv/bin/activate
./scripts/patch_vllm.sh
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

`uv sync` installs the locked Python dependencies. The scripts separately patch
the supported vLLM wheel for WSL, build the pinned whisper.cpp CLI for the AMD
GPU reported by `rocminfo`, download and verify the Whisper and VAD weights, and
check the completed environment. uv does not compile whisper.cpp or download
model weights.

## Audio Extraction

The base environment is sufficient to exercise FFmpeg audio extraction:

```bash
uv run --managed-python python -m src.audio /path/to/video.mkv -o output.wav
```

The output is 16 kHz, mono, 16-bit PCM WAV for downstream ASR processing.

## Full Pipeline

The full pipeline uses the managed whisper.cpp backend:

```bash
python run.py /path/to/video.mp4
python run.py /path/to/movie.mkv --output-dir ./subs --src-lang de --lang French
python run.py /path/to/video.mp4 --keep-temp
python run.py /path/to/video.mp4 --translated-only
```

Useful options:

- `--src-lang`: source language code; whisper.cpp auto-detects it when omitted
- `--keep-temp`: preserve intermediate WAV and SRT files
- `--force`: overwrite output and rebuild intermediates
- `--translated-only`: omit source text from the final subtitles

## Standalone Transcription

Transcribe a prepared audio file with automatic source-language detection or an
explicit language code:

```bash
python -m src.transcribe /path/to/audio.wav -o ./subs
python -m src.transcribe /path/to/audio.wav -o ./subs -l de
python -m src.transcribe /path/to/audio.wav -o ./subs --quiet
```

The command displays native transcription progress by default. `--quiet` hides
the Python progress bar without changing transcription behavior.

## Subtitle Cleanup

Run conservative SRT cleanup independently with:

```bash
python -m src.clean /path/to/subtitles.srt -o cleaned.srt
```

Cleanup trims surrounding whitespace, removes empty or punctuation-only cues,
and removes cues made from four or more identical effective characters. Two
identical consecutive cues are merged when their gap is at most 250 ms; close
runs of three or more identical cues are discarded. Other content is preserved.

## Managed whisper.cpp ASR Profile

The project intentionally exposes one fixed ASR profile rather than public
model or device selection. Its managed assets are:

```text
.venv/bin/whisper-cli
models/whisper/ggml-large-v3-turbo.bin
models/whisper/ggml-silero-v6.2.0.bin
```

The quality-oriented decoder and VAD settings are:

```text
model: large-v3-turbo
processors: 1
beam size: 5
max context: 48
VAD threshold: 0.15
VAD minimum speech: 250 ms
VAD minimum silence: 120 ms
VAD maximum speech: 30 s
VAD speech padding: 250 ms
VAD sample overlap: 0 s
```

These settings are internal and fixed.

## Development

Development tools are declared in the `dev` group and are not installed by the
default sync. Create a development environment with:

```bash
uv sync --locked --managed-python --group dev
source .venv/bin/activate
```

Run the project checks from the activated environment:

```bash
pytest
ruff check .
```

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
weights are repository-managed files under `models/whisper` and are prepared by
`scripts/download_weights.sh`.
