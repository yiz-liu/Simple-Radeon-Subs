# Development Guidelines

## Scope

Simple Radeon Subs generates and translates movie subtitles locally on AMD
Radeon GPUs under WSL. Keep changes focused on the fixed, tested runtime instead
of adding speculative backends or configuration surfaces.

## Environment

Create or update `.venv` from the committed lock:

```bash
# Base runtime
uv sync --locked --managed-python

# Development tools
uv sync --locked --managed-python --group dev

# Complete GPU runtime
uv sync --locked --managed-python --group dev --group vllm
```

After `.venv` exists, activate it before running repository commands:

```bash
source .venv/bin/activate
```

After dependency declarations change, regenerate and commit the lock:

```bash
uv lock --managed-python
```

Dependency ownership is explicit:

- the operating system provides ROCm, FFmpeg, CMake, and Git;
- uv manages Python 3.12 and declared Python packages;
- project scripts patch vLLM, build whisper.cpp, and download weights.

Prepare the complete native runtime with:

```bash
./.venv/bin/python scripts/patch_vllm.py
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

Do not install undeclared Python packages directly into `.venv`.

## Commands

Run the full pipeline:

```bash
python run.py /path/to/video.mp4
python run.py /path/to/movie.mkv --output-dir ./subs --src-lang de --lang French
python run.py /path/to/video.mp4 --keep-temp
python run.py /path/to/videos/
```

Run individual stages:

```bash
python -m src.audio /path/to/video.mp4 -o output.wav
python -m src.transcribe /path/to/audio.wav -o ./subs
python -m src.transcribe /path/to/audio.wav -o ./subs -l de
python -m src.clean /path/to/subs.srt -o cleaned.srt
python -m src.translate /path/to/subs.srt -o translated.srt --lang Chinese
```

Run development checks:

```bash
pytest
ruff check .
basedpyright --level error run.py src tests
```

`basedpyright` is an optional external developer tool. It is not a project
runtime dependency.

## Architecture

- `run.py`: CLI input discovery and pipeline entry point
- `src/pipeline.py`: stage-wise batch scheduling, sidecar reuse, and failure isolation
- `src/audio.py`: FFmpeg audio extraction and progress parsing
- `src/transcribe.py`: standalone fixed whisper.cpp adapter
- `src/whisper_batch.py`: one-process multi-file whisper.cpp execution
- `src/clean.py`: conservative SRT normalization and duplicate filtering
- `src/translate.py`: standalone translation CLI
- `src/translation.py`: subtitle preparation and local vLLM inference
- `src/translation_requests.py`: request windows and structured output parsing
- `src/translation_output.py`: final subtitle construction and atomic publication
- `src/config.py`: managed paths and fixed runtime constants
- `src/logger.py`: shared logging configuration

Directory inputs run stage by stage across all pending files. A final SRT skips
the input before native runtimes initialize. Intermediate state is stored in a
per-input sidecar and reused after failure; successful sidecars are removed
unless `--keep-temp` is active.

## Implementation Rules

- Target Python 3.12. Use built-in generics and `T | None` syntax.
- Type all public functions and methods. Avoid untyped dictionaries at module
  boundaries.
- Use `pathlib.Path` for filesystem paths and resolve external inputs early.
- Order imports as standard library, third-party packages, then absolute local
  imports.
- Keep fixed managed paths and runtime constants in `src/config.py`.
- Use `src.logger.logger`; include relevant input or stage context in messages.
- Catch errors at I/O and native-runtime boundaries. Preserve causes and report
  subprocess return codes and useful stderr context.
- Use `argparse` for command-line entry points and document every public option.
- Use `pysrt` for SRT handling. Preserve timestamps during text-only changes,
  re-index after structural changes, and save with UTF-8 encoding. Publish final
  translation output atomically.
- Require system `ffmpeg` and `ffprobe`. Extract 16 kHz mono 16-bit PCM WAV and
  consume machine-readable progress output.
- Keep the managed whisper.cpp and vLLM profiles fixed unless measurements
  justify changing the project baseline.

## Testing

Tests live under `tests/` and run with pytest. Add a focused regression test for
every behavior change.

Prefer unit tests that replace native-process boundaries and do not require a
GPU. Use `tests/example.mp4` and `tests/example_en.mp3` only when real media is
needed to cover an adapter boundary. Validate actual ROCm, whisper.cpp, or vLLM
execution manually when the native integration changes.

## Verified Runtime

- Python 3.12
- WSL2 on x86-64 Linux
- AMD Radeon RX 9070 XT (`gfx1201`)
- system ROCm 7.2.4
- vLLM 0.25.1 `rocm723` wheel stack

Other compatible AMD GPUs may work but are outside the current tested baseline.
