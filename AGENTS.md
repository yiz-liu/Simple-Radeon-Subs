# AGENTS.md - Development Guidelines for Simple Radeon Subs

## Project Overview
Movie subtitle generation and local translation tool targeted at AMD GPUs on WSL with ROCm.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for virtual environment and dependency management. Before executing any commands, please:

```bash
source .venv/bin/activate
```

### Running the Application
```bash
# Full pipeline (extract -> transcribe -> clean -> translate)
python run.py /path/to/video.mp4

# With custom output directory and explicit source and target languages
python run.py /path/to/movie.mkv --output-dir ./subs --src-lang de --lang French

# Keep temporary files for debugging
python run.py /path/to/video.mp4 --keep-temp

# Process entire directory of videos
python run.py /path/to/videos/
```

### Running Individual Modules
```bash
# Extract audio only
python -m src.audio /path/to/video.mp4 -o output.wav

# Transcribe audio only
python -m src.transcribe /path/to/audio.wav -o ./subs

# Transcribe audio with an explicit language code
python -m src.transcribe /path/to/audio.wav -o ./subs -l de

# Clean subtitles only
python -m src.clean /path/to/subs.srt -o cleaned.srt

# Translate subtitles only
python -m src.translate /path/to/subs.srt -o translated.srt --lang Chinese
```

### Environment Setup
```bash
# Install the project Python version
uv python install 3.12

# Generate the lock file after dependency changes
uv lock --managed-python

# Create the development environment from the committed lock file
uv sync --locked --managed-python --group dev --group vllm

# Activate virtual environment
source .venv/bin/activate

# Prepare and verify the managed native ASR runtime
./scripts/patch_vllm.sh
./scripts/install_whisper_cli.sh
./scripts/download_weights.sh
./scripts/doctor.sh
```

The ASR executable and weights are installed separately from uv-managed Python
dependencies. Omit the source language to use whisper.cpp auto-detection.

## Code Style Guidelines

### Type Hints
- Use Python 3.12+ union syntax: `str | Path` instead of `Union[str, Path]`
- Always include return types for public methods
- Use `Optional[T]` for nullable parameters
- Import typing from standard library: `from typing import Optional, List, Dict`

### File Paths
- Always use `pathlib.Path` instead of string paths
- Resolve paths immediately: `Path(input_path).resolve()`
- Use `Path.parent`, `Path.stem`, `Path.suffix` for path manipulation
- Create directories with `mkdir(parents=True, exist_ok=True)`

### Import Organization
1. Standard library imports
2. Third-party imports
3. Local imports (use absolute imports: `from src.X import Y`)

### Naming Conventions
- Classes: PascalCase (`AudioExtractor`, `Transcriber`)
- Functions/Methods: snake_case (`clean_srt`, `process_video`)
- Variables: snake_case (`audio_path`, `target_lang`)
- Constants: UPPER_SNAKE_CASE (`TRANSLATION_MODEL_PATH`, `AUDIO_SAMPLE_RATE`)
- Private methods: underscore prefix (`_load_model`, `_resolve_ffmpeg`)

### Error Handling
- Use try/except for I/O operations and native runtime boundaries
- Log errors with `logger.error()` including exception details: `exc_info=True`
- Raise descriptive exceptions with context: `FileNotFoundError(f"Input not found: {path}")`
- Handle subprocess failures with returncode checks

### Logging
- Import from `src.logger`: `from src.logger import logger`
- Use appropriate levels: `logger.info()`, `logger.warning()`, `logger.error()`
- Include relevant context in log messages
- Use `logger.info()` for progress updates, `logger.warning()` for non-critical issues

### Classes and Methods
- Include docstrings for all classes and public methods
- Keep methods focused and single-purpose
- Use type hints for all parameters and return values

### CLI Arguments
- Use `argparse` for command-line interfaces
- Provide help text for all arguments
- Use sensible defaults for optional parameters
- Support both file and directory inputs where appropriate

### Configuration
- Store fixed managed paths and runtime constants in `src/config.py`
- Define constants at module level

### Subtitle Processing
- Use `pysrt` library for SRT file handling
- Preserve timestamps during text transformations
- Re-index subtitles after filtering/merging
- Save with UTF-8 encoding

### Audio Processing
- Target format: 16kHz, Mono, 16-bit PCM WAV
- Use FFmpeg for audio extraction
- Parse FFmpeg progress output for progress bars
- Require system FFmpeg and ffprobe binaries on `PATH`

### Testing Notes
- Pytest is configured for tests under `tests/`
- Manual testing required for each module
- Test with various video formats and languages
- Verify GPU acceleration is working

## Architecture Notes

### Module Responsibilities
- `run.py`: Orchestration and CLI entry point
- `src/audio.py`: Audio extraction with FFmpeg
- `src/transcribe.py`: Fixed whisper.cpp adapter using `.venv/bin/whisper-cli` and managed weights under `models/whisper`
- `src/clean.py`: Conservative SRT normalization and duplicate filtering
- `src/translate.py`: CLI for the fixed local vLLM translation backend
- `src/translation.py`: Subtitle batching and local vLLM inference
- `src/config.py`: Managed paths and runtime constants
- `src/logger.py`: Logging setup

### Data Flow
1. Video file → Audio extraction (WAV)
2. WAV → Whisper transcription (SRT)
3. SRT → Conservative cleanup (trim, safe filtering, time-aware deduplication)
4. Cleaned SRT → local vLLM translation (batched inference)
5. Final SRT output

### Performance Considerations
- ASR runs through the fixed HIP whisper.cpp profile with built-in VAD and native progress reporting
- Translation submits all subtitle batches to the local vLLM scheduler
- Batch size for translation: 16 lines per prompt
- FFmpeg extraction shows real-time progress

## Hardware Requirements
- AMD GPU with a compatible ROCm release
- Python 3.12+
- Sufficient GPU memory for the managed Whisper and translation models
