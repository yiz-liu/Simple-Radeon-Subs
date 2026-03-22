# AGENTS.md - Development Guidelines for Simple Radeon Subs

## Project Overview
Movie subtitle translation tool using OpenAI Whisper (transcription) and Google Gemini (translation). Optimized for AMD GPUs on WSL with ROCm 7.2.0.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for virtual environment and dependency management. Before executing any commands, please:

```bash
source .venv/bin/activate
```

### Running the Application
```bash
# Full pipeline (extract -> transcribe -> clean -> translate)
python run.py /path/to/video.mp4

# With custom output directory and language
python run.py /path/to/movie.mkv --output-dir ./subs --lang ja

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
python -m src.transcribe /path/to/audio.wav -o ./subs -m large-v3-turbo

# Clean subtitles only
python -m src.clean /path/to/subs.srt -o cleaned.srt

# Translate subtitles only
python -m src.translate /path/to/subs.srt -o translated.srt --lang Chinese
```

### Environment Setup
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install ROCm-specific PyTorch (REQUIRED - do NOT use PyPI)
uv pip install torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cw312-cw312-linux_x86_64.whl
uv pip install triton-3.5.1+rocm7.2.0.gita272dfa8-cw312-cw312-linux_x86_64.whl

# Verify GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

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
- Constants: UPPER_SNAKE_CASE (`GEMINI_API_KEY`, `AUDIO_SAMPLE_RATE`)
- Private methods: underscore prefix (`_load_model`, `_resolve_ffmpeg`)

### Error Handling
- Use try/except for I/O operations and external API calls
- Log errors with `logger.error()` including exception details: `exc_info=True`
- Raise descriptive exceptions with context: `FileNotFoundError(f"Input not found: {path}")`
- Handle subprocess failures with returncode checks

### Logging
- Import from `src.logger`: `from src.logger import logger`
- Use appropriate levels: `logger.info()`, `logger.warning()`, `logger.error()`
- Include relevant context in log messages
- Use `logger.info()` for progress updates, `logger.warning()` for non-critical issues

### Concurrency
- Use `concurrent.futures.ThreadPoolExecutor` for parallel API calls
- Use `tqdm` for progress bars in concurrent operations
- Implement retry logic with exponential backoff for API calls
- Use `as_completed()` to process results as they finish

### Classes and Methods
- Include docstrings for all classes and public methods
- Use lazy loading for heavy resources (e.g., Whisper models)
- Keep methods focused and single-purpose
- Use type hints for all parameters and return values

### CLI Arguments
- Use `argparse` for command-line interfaces
- Provide help text for all arguments
- Use sensible defaults for optional parameters
- Support both file and directory inputs where appropriate

### Configuration
- Store configuration in `src/config.py`
- Use `python-dotenv` for environment variables
- Load `.env` from project root
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
- Handle both system and local FFmpeg binaries

### API Integration
- Implement retry logic for HTTP requests
- Use timeouts for all API calls
- Handle rate limiting (HTTP 429) with backoff
- Validate API responses before processing

### Testing Notes
- No automated test framework currently configured
- Manual testing required for each module
- Test with various video formats and languages
- Verify GPU acceleration is working

## Architecture Notes

### Module Responsibilities
- `run.py`: Orchestration and CLI entry point
- `src/audio.py`: Audio extraction with FFmpeg
- `src/transcribe.py`: Whisper-based transcription
- `src/clean.py`: Subtitle text cleaning and filtering
- `src/translate.py`: Gemini API translation with concurrency
- `src/config.py`: Configuration and environment variables
- `src/logger.py`: Logging setup

### Data Flow
1. Video file → Audio extraction (WAV)
2. WAV → Whisper transcription (SRT)
3. SRT → Text cleaning (remove SDH, HTML, hallucinations)
4. Cleaned SRT → Gemini translation (concurrent batches)
5. Final SRT output

### Performance Considerations
- Whisper models are loaded lazily to save memory
- Translation uses concurrent API calls (default: 10 workers)
- Batch size for translation: 30 lines per request
- FFmpeg extraction shows real-time progress
- Use fp16=True for GPU acceleration when available

## Environment Variables
Required in `.env`:
- `GEMINI_API_KEY`: Google Gemini API for translation
- `GEMINI_API_URL`: Optional custom API endpoint

## Hardware Requirements
- AMD GPU with ROCm 7.2.0 support
- Python 3.12+
- Sufficient GPU memory for Whisper model (large-v3-turbo recommended)
