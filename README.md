# Simple Radeon Subs 🎬

**Simple subtitle maker for Radeon GPUs on WSL.**

A One-Time-Code simple toolchain for automatically generating movie subtitles. This project leverages **OpenAI Whisper** for transcription and **Google Gemini** for high-quality, concurrent translation, specifically optimized for **AMD GPUs** running on WSL with **ROCm 7.2.0**.

> 📖 **Curious about the journey?** Read [DEVLOG.md](DEVLOG.md) to see how we navigated the "Wild West" of AMD ROCm on WSL, failed with CTranslate2, and succeeded with native PyTorch.

## 🚀 Steps

- **Audio Extraction**: Extracts optimized audio (16kHz, Mono) from video files using a portable FFmpeg build.
- **Transcription**: Uses Silero VAD to pre-segment speech, then runs `openai-whisper` (accelerated by ROCm/PyTorch) with `clip_timestamps` to generate timestamped SRT files.
- **Cleaning**: Automatically removes SDH tags (e.g., `[Music]`, `(Applause)`), HTML tags, and common ASR hallucinations.
- **Translation**: Translates subtitles into your target language (default: Chinese) using the Google Gemini API with multi-threaded concurrency for speed.

## 🛠️ Prerequisites

- **OS**: Linux (Tested on WSL with ROCm 7.2.0)
- **Hardware**: AMD GPU (Tested on Radeon RX 9070 XT, gfx1201)
- **Python**: 3.12+
- **API Key**: A Google Gemini API Key

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yiz-liu/Simple-Radeon-Subs.git
cd Simple-Radeon-Subs
```

### 2. Set up Virtual Environment & Install Dependencies
This project uses [uv](https://docs.astral.sh/uv/) for virtual environment and dependency management.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Install ROCm-PyTorch (Crucial!)
⚠️ **PITFALL ALERT**: Do not install `torch` from PyPI.

You must install the pre-built wheels specifically built for your ROCm version (e.g., ROCm 7.2.0).

Please see [Install Radeon software for WSL with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html) and [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html)

```bash
# Example command (adjust filenames as needed)
uv pip install torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
uv pip install triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

*Verify installation:*
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Should print: CUDA Available: True
```

### 4. (Optional) Setup Portable FFmpeg
If you have installed FFmpeg, just go with it. To avoid system conflicts (and as a control freak), I choose a static build of FFmpeg.

```bash
mkdir -p tools
# Download static build (example URL, check for latest)
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz -O tools/ffmpeg.tar.xz
tar -xJf tools/ffmpeg.tar.xz -C tools/
mv tools/ffmpeg-*-amd64-static tools/ffmpeg
rm tools/ffmpeg.tar.xz
```

### 5. Environment Configuration
Create a `.env` file in the project root:

```bash
cp .env.template .env
# Edit .env and add your GEMINI_API_KEY
```

## 🏃 Usage

Run the full pipeline (Extract -> Transcribe -> Clean -> Translate) with `run.py`.

```bash
# Basic usage with the default VAD + Whisper pipeline
python run.py /path/to/video.mp4

# Specify output directory and target language
python run.py /path/to/movie.mkv --output-dir ./subs --lang "French"

# Keep intermediate files after success
python run.py /path/to/video.mp4 --keep-temp

# Disable VAD and transcribe the full audio with Whisper
python run.py /path/to/video.mp4 --no-vad

# Overwrite an existing generated subtitle file and restart from clean intermediates
python run.py /path/to/video.mp4 -f

# Generate translated subtitles only instead of bilingual output
python run.py /path/to/video.mp4 --translated-only
```

### Arguments
- `input`: Path to the input video or audio file.
- `-o, --output-dir`: Directory to save the final `.srt` file.
- `--lang`: Target language (default: "Chinese").
- `--src-lang`: Source language of the audio (e.g., 'en', 'zh'). Auto-detects if omitted.
- `--model`: Whisper model to use (default: `large-v3-turbo`).
- `--keep-temp`: Keep intermediate files (`.wav`, `.vad.json`, raw `.srt`, `.cleaned.srt`) after successful completion.
- `-f, --force`: Overwrite an existing generated subtitle file and delete old intermediate files before restarting.
- `--translated-only`: Output only translated subtitles instead of the default bilingual subtitles.
- `--no-vad`: Disable Silero VAD pre-segmentation and let Whisper transcribe the full audio directly.

By default, the full pipeline now runs Silero VAD before Whisper. The generated `.vad.json` report contains the speech segments and flattened `clip_timestamps` that are passed into Whisper.

When VAD is enabled, transcription progress is now shown against detected speech coverage instead of the full media duration, so long silent gaps no longer leave the Whisper progress bar looking unfinished at the end.

For standalone VAD analysis, run `python -m src.vad /path/to/audio.wav`. For standalone transcription, `python -m src.transcribe` still shows a progress bar by default, supports `--clip-timestamps-file`, uses the same clip-aware progress behavior when timestamps are provided, and accepts `--verbose-text` or `--quiet` just like before.

By default, translated subtitle output is bilingual: each saved subtitle block contains the cleaned source text followed by the translated text. Entries whose translated text is empty or skipped are removed entirely so the final bilingual subtitles stay aligned with the filtered translation result.

By default, intermediate files are now written to the final output directory so interrupted runs can reuse them later. They are only deleted after the final subtitle is generated successfully; if a run fails midway, the intermediate files stay in place for the next retry.

## 🛑 Troubleshooting & Pitfalls

### "FFMPEG_MISSING" or "FileNotFoundError"
- Ensure `tools/ffmpeg/ffmpeg` exists and is executable.
- The script looks for FFmpeg in `tools/ffmpeg/` relative to the project root.

### CTranslate2 / Faster-Whisper Issues
- **Why standard Whisper?** We attempted to use `CTranslate2` for faster inference, but compilation failed due to deep incompatibilities with **ROCm 7.2.0** (Thrust/hipBLAS API changes).
- **Solution**: We reverted to the official `openai-whisper` package using the native PyTorch ROCm wheels, which works perfectly.

### Translation Alignment
- If the translation count doesn't match the source lines, the script will output a warning or fill gaps. We use strict prompting ("No Bullshit", "Strict Alignment") to minimize this, but Gemini can occasionally be creative.
