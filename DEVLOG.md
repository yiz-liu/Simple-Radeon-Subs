# The Journey: Building Simple Radeon Subs on WSL

> **TL;DR:** This document chronicles the development journey of **Simple Radeon Subs**, specifically focusing on the challenges of running AI inference on AMD Radeon GPUs within a WSL (Windows Subsystem for Linux) environment. It documents the failures, the "aha!" moments, and the final working configuration.

---

## Part 1: The Ambition

The goal was simple: Create a high-performance, local movie subtitle translator.
The hardware: **AMD Radeon RX 9070 XT**.
The environment: **WSL2 (Ubuntu)**.

While Nvidia users enjoy a smooth sailing with CUDA, the AMD ROCm ecosystem on consumer cards (especially on Windows/WSL) is a bit of a wild west.

## Part 2: The "Faster-Whisper" Trap (Failure)

My first instinct was to use [CTranslate2](https://github.com/OpenNMT/CTranslate2) (the engine behind `faster-whisper`) because, well, it's *faster*. I found a fork `CTranslate2-rocm` and attempted to build it from source.

### The Setup
- **System**: Linux (ROCm 7.2.0, gfx1201)
- **Repo**: `https://github.com/arlo-phoenix/CTranslate2-rocm.git`
- **Compiler**: AMD clang++ (ROCm 7.2.0)

### The Struggle
1.  **CMake Hell**: The build system required specific CMake versions and policy tweaks just to start configuration.
2.  **Missing Libraries**: `libiomp5` (Intel OpenMP) issues popped up, requiring manual flags to force `libomp`.
3.  **The Wall**: After hours of configuring, the build failed during compilation with errors like:
    ```
    error: no template named 'counting_iterator' in namespace 'thrust'
    error: use of undeclared identifier 'hipblasGemmEx_v2'
    ```

### The Lesson
**ROCm 7.2.0 broke backward compatibility.**
The `hipBLAS` and `Thrust` libraries in the newer ROCm versions have refactored APIs that older CTranslate2 forks rely on. Fixing this would require rewriting the CUDA-to-HIP compatibility layer—a massive undertaking.

**Status:** ❌ Abandoned.

## Part 3: The Pivot to Native PyTorch (Success)

If "faster" wasn't an option, maybe "standard" was. I decided to try the official `openai-whisper` package, but backed by a custom-built PyTorch for ROCm.

### The Breakthrough
Instead of `pip install torch`, which pulls the standard CUDA/CPU version, I found the specific wheels for ROCm 7.2.0.

It worked! The official Whisper implementation (`large-v3-turbo`) ran smoothly on the RX 9070 XT, utilizing the GPU acceleration via the ROCm-PyTorch backend.

## Part 4: The Toolchain Design

With the core inference engine working, I designed the rest of the pipeline:

1.  **Audio Extraction**:
    -   **Problem**: System FFmpeg can be messy or missing.
    -   **Solution**: Download a **static FFmpeg binary** into `tools/ffmpeg/`. It's portable, version-controlled, and doesn't require `sudo`.

2.  **Transcription**:
    -   **Engine**: `openai-whisper`.
    -   **Model**: `large-v3-turbo` (Good balance of speed/accuracy).
    -   **Format**: Output strictly to SRT.

3.  **Cleaning**:
    -   **Problem**: Whisper hallucinations ("Subtitle by Amara.org") and SDH tags (`[Music]`).
    -   **Solution**: A custom regex-based cleaner (`src/clean.py`) to strip these out before translation.

4.  **Translation**:
    -   **Engine**: **Google Gemini API**.
    -   **Strategy**: Concurrency. Translating line-by-line is too slow. I built a threaded translator (`src/translate.py`) that sends batches of subtitles in parallel.
    -   **Prompt Engineering**: "Strict Alignment" and "No Bullshit" rules to keep Gemini focused.

## Conclusion

Building **Simple Radeon Subs** taught me that while the AMD ROCm ecosystem is improving, it still requires navigating a maze of version incompatibilities.

- **Don't** try to compile legacy CUDA projects on bleeding-edge ROCm unless you love C++ error logs.
- **Do** use official PyTorch wheels for ROCm; they are surprisingly robust.
- **WSL** is a viable platform for AI on AMD, provided you get the drivers right.

Enjoy the tool!
