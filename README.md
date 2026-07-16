# Simple Radeon Subs

Generate, clean, and translate movie subtitles on AMD Radeon GPUs under WSL.

The pipeline is:

```text
video -> FFmpeg audio extraction -> VAD -> ASR -> subtitle cleanup -> translation
```

The current uv configuration intentionally contains only the lightweight
dependencies required to develop and test audio extraction. The ROCm ASR stack
will be added after the ASR backend and benchmark plan are finalized.

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

## Audio Extraction

The base environment is sufficient to exercise FFmpeg audio extraction:

```bash
uv run --managed-python python -m src.audio /path/to/video.mkv -o output.wav
```

The output is 16 kHz, mono, 16-bit PCM WAV for downstream ASR processing.

## Full Pipeline

The following commands require the future ROCm ASR dependency group in addition
to the base environment:

```bash
uv run --managed-python python run.py /path/to/video.mp4
uv run --managed-python python run.py /path/to/movie.mkv --output-dir ./subs --lang Chinese
uv run --managed-python python run.py /path/to/video.mp4 --keep-temp
uv run --managed-python python run.py /path/to/video.mp4 --no-vad
uv run --managed-python python run.py /path/to/video.mp4 --translated-only
```

Useful options:

- `--src-lang`: source language; auto-detected when omitted
- `--model`: ASR model name
- `--keep-temp`: preserve intermediate WAV, VAD, and SRT files
- `--force`: overwrite output and rebuild intermediates
- `--translated-only`: omit source text from the final subtitles
- `--no-vad`: bypass VAD and transcribe the complete audio

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

ROCm, the AMD driver, and model caches are system or machine resources and are
not managed by uv.
