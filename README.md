# Simple Radeon Subs 🎬

**Simple subtitle maker for Radeon GPUs on WSL.**

A One-Time-Code simple toolchain for automatically generating movie subtitles. This project leverages **OpenAI Whisper** for transcription and **Google Gemini** for high-quality, concurrent translation, specifically optimized for **AMD GPUs** running on WSL with **ROCm 7.2.0**.

> 📖 **Curious about the journey?** Read [DEVLOG.md](DEVLOG.md) to see how we navigated the "Wild West" of AMD ROCm on WSL, failed with CTranslate2, and succeeded with native PyTorch.

## 🚀 Steps

- **Audio Extraction**: Extracts optimized audio (16kHz, Mono) from video files using a portable FFmpeg build.
- **Transcription**: Uses `openai-whisper` (accelerated by ROCm/PyTorch) to generate timestamped SRT files.
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

### 2. Set up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install ROCm-PyTorch (Crucial!)
⚠️ **PITFALL ALERT**: Do not install `torch` from PyPI.

You must install the pre-built wheels specifically built for your ROCm version (e.g., ROCm 7.2.0).

Please see [Install Radeon software for WSL with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html) and [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html)

```bash
# Example command (adjust filenames as needed)
pip install torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
pip install triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

*Verify installation:*
```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Should print: CUDA Available: True
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Setup Portable FFmpeg
If you have installed FFmpeg, just go with it. To avoid system conflicts (and as a control freak), I choose a static build of FFmpeg.

```bash
mkdir -p tools
# Download static build (example URL, check for latest)
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz -O tools/ffmpeg.tar.xz
tar -xJf tools/ffmpeg.tar.xz -C tools/
mv tools/ffmpeg-*-amd64-static tools/ffmpeg
rm tools/ffmpeg.tar.xz
```

### 6. Environment Configuration
Create a `.env` file in the project root:

```bash
cp .env.template .env
# Edit .env and add your GEMINI_API_KEY
```

## 🏃 Usage

Run the full pipeline (Extract -> Transcribe -> Clean -> Translate) with `run.py`.

```bash
# Basic usage (defaults to Chinese translation)
python run.py /path/to/video.mp4

# Specify output directory and target language
python run.py /path/to/movie.mkv --output-dir ./subs --lang "French"

# Keep temporary files (wav, raw srt) for debugging
python run.py /path/to/video.mp4 --keep-temp
```

### Arguments
- `input`: Path to the input video or audio file.
- `-o, --output-dir`: Directory to save the final `.srt` file.
- `--lang`: Target language (default: "Chinese").
- `--src-lang`: Source language of the audio (e.g., 'en', 'zh'). Auto-detects if omitted.
- `--model`: Whisper model to use (default: `large-v3-turbo`).
- `--keep-temp`: Don't delete intermediate files (`.wav`, `.cleaned.srt`).

## 🛑 Troubleshooting & Pitfalls

### "FFMPEG_MISSING" or "FileNotFoundError"
- Ensure `tools/ffmpeg/ffmpeg` exists and is executable.
- The script looks for FFmpeg in `tools/ffmpeg/` relative to the project root.

### CTranslate2 / Faster-Whisper Issues
- **Why standard Whisper?** We attempted to use `CTranslate2` for faster inference, but compilation failed due to deep incompatibilities with **ROCm 7.2.0** (Thrust/hipBLAS API changes).
- **Solution**: We reverted to the official `openai-whisper` package using the native PyTorch ROCm wheels, which works perfectly.

### Translation Alignment
- If the translation count doesn't match the source lines, the script will output a warning or fill gaps. We use strict prompting ("No Bullshit", "Strict Alignment") to minimize this, but Gemini can occasionally be creative.
