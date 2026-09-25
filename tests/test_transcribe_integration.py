import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from src.audio import AudioExtractor
from src.config import (
    WHISPER_CLI_PATH,
    WHISPER_MODEL_PATH,
    WHISPER_VAD_MODEL_PATH,
)
from src.transcribe import Transcriber, TranscriptionTask

EXAMPLE_VIDEO: Final = Path(__file__).with_name("example.mp4")
EXAMPLE_AUDIO: Final = Path(__file__).with_name("example_en.mp3")
REAL_ASR_AVAILABLE: Final = (
    WHISPER_CLI_PATH.is_file()
    and os.access(WHISPER_CLI_PATH, os.X_OK)
    and WHISPER_MODEL_PATH.is_file()
    and WHISPER_VAD_MODEL_PATH.is_file()
)
SRT_VALIDATION_SCRIPT: Final = """
import sys
from itertools import pairwise

import pysrt

subtitles = pysrt.open(sys.argv[1], encoding="utf-8")
assert subtitles
assert any(subtitle.text.strip() for subtitle in subtitles)
assert [subtitle.index for subtitle in subtitles] == list(range(1, len(subtitles) + 1))
assert all(subtitle.start.ordinal < subtitle.end.ordinal for subtitle in subtitles)
assert all(
    previous.end.ordinal <= current.start.ordinal
    for previous, current in pairwise(subtitles)
)
print(f"validated {len(subtitles)} subtitles")
"""


def test_module_cli_exposes_only_the_fixed_adapter_options() -> None:
    # Given: The transcribe module's public command-line surface.

    # When: A user asks for help.
    completed = subprocess.run(
        [sys.executable, "-m", "src.transcribe", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: Only fixed adapter controls, including the VAD opt-out, remain.
    assert completed.returncode == 0
    assert all(
        flag in completed.stdout
        for flag in ("--output-dir", "--language", "--quiet", "--disable-vad")
    )
    assert all(
        flag not in completed.stdout
        for flag in ("--model", "--device", "--verbose-text", "--enable-vad")
    )


def test_pipeline_cli_does_not_expose_asr_model_selection() -> None:
    # Given: The end-to-end pipeline's public command-line surface.

    # When: A user asks for help.
    completed = subprocess.run(
        [sys.executable, "run.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: The fixed ASR model is not a user-selectable option.
    assert completed.returncode == 0
    assert "--model" not in completed.stdout
    assert "--disable-vad" in completed.stdout
    assert "--enable-vad" not in completed.stdout


def test_module_cli_rejects_a_missing_audio_file_without_creating_an_srt(
    tmp_path: Path,
) -> None:
    # Given: A missing WAV path and a fresh subtitle directory.
    missing_audio = tmp_path / "missing.wav"
    output_dir = tmp_path / "subtitles"

    # When: The public module CLI receives the malformed input path.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(missing_audio),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: The CLI reports failure without publishing a subtitle.
    assert completed.returncode != 0
    assert not (output_dir / "missing.srt").exists()


def test_module_cli_reports_an_invalid_output_directory_without_a_traceback(
    tmp_path: Path,
) -> None:
    # Given: A valid audio path and a regular file where a directory is required.
    audio_path = tmp_path / "audio.wav"
    output_file = tmp_path / "not-a-directory"
    _ = audio_path.touch()
    _ = output_file.touch()

    # When: The public CLI reaches the filesystem output boundary.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(audio_path),
            "--output-dir",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Then: It reports a controlled failure without publishing an SRT.
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "audio.srt").exists()


@pytest.mark.skipif(
    not REAL_ASR_AVAILABLE,
    reason="managed whisper.cpp CLI, full model, or VAD model is unavailable",
)
def test_module_cli_transcribes_the_real_media_fixture_on_rocm(tmp_path: Path) -> None:
    # Given: The shared real-media fixture converted through the production extractor.
    audio_path = AudioExtractor().extract(EXAMPLE_VIDEO, tmp_path / "example.wav")
    output_dir = tmp_path / "subtitles"

    # When: The public module CLI transcribes through managed whisper.cpp.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.transcribe",
            str(audio_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Then: Visible completion accompanies a well-formed, ordered subtitle timeline.
    subtitle_path = output_dir / "example.srt"
    assert completed.returncode == 0, completed.stderr
    assert "Transcribing" in completed.stderr
    assert "100%" in completed.stderr
    validation = subprocess.run(
        [sys.executable, "-c", SRT_VALIDATION_SCRIPT, str(subtitle_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert validation.returncode == 0, validation.stderr
    assert validation.stdout.startswith("validated ")


@pytest.mark.skipif(
    not REAL_ASR_AVAILABLE or not EXAMPLE_AUDIO.is_file(),
    reason="managed whisper.cpp runtime or the second media fixture is unavailable",
)
def test_transcribe_many_processes_two_real_audio_files_in_one_batch(
    tmp_path: Path,
) -> None:
    # Given: The video fixture converted to WAV and the independent MP3 fixture.
    first_audio = AudioExtractor().extract(EXAMPLE_VIDEO, tmp_path / "first.wav")
    tasks = (
        TranscriptionTask(first_audio, tmp_path / "first.srt"),
        TranscriptionTask(EXAMPLE_AUDIO, tmp_path / "second.srt"),
    )

    # When: Both files enter one managed whisper.cpp invocation.
    failures = Transcriber().transcribe_many(tasks, language="en", quiet=True)

    # Then: Both outputs contain valid ordered subtitle timelines.
    assert failures == []
    for task in tasks:
        validation = subprocess.run(
            [sys.executable, "-c", SRT_VALIDATION_SCRIPT, str(task.output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert validation.returncode == 0, validation.stderr


def _download_script(tmp_path: Path) -> Path:
    import shutil

    script = tmp_path / "scripts" / "download_weights.sh"
    script.parent.mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/download_weights.sh", script
    )
    return script


def test_weight_check_does_not_download_missing_files(tmp_path: Path) -> None:
    script = _download_script(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--asr-backend", "qwen", "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing" in result.stderr
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("check_only", [False, True])
def test_weight_preparation_preserves_unexpected_existing_content(
    tmp_path: Path, check_only: bool
) -> None:
    script = _download_script(tmp_path)
    model = tmp_path / "models/Qwen3-ASR-1.7B/chat_template.json"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"unexpected existing content")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--asr-backend",
            "qwen",
            *(["--check"] if check_only else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert model.read_bytes() == b"unexpected existing content"
    assert "Refusing to overwrite" in result.stderr
    assert not list((tmp_path / "models").glob(".download.*"))


def test_weight_download_rejects_corruption_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _download_script(tmp_path)
    _fake_hf(tmp_path, monkeypatch, "corrupt download")
    result = subprocess.run(
        ["bash", str(script), "--asr-backend", "qwen"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "failed checksum verification" in result.stderr
    assert not (tmp_path / "models/Qwen3-ASR-1.7B/chat_template.json").exists()
    assert not list((tmp_path / "models").glob(".download.*"))


def _fake_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> Path:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "uvx"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
assert args[:2] == ["hf", "download"]
assert args[3] == "--revision"
files = ["chat_template.json", "config.json", "generation_config.json",
         "merges.txt", "preprocessor_config.json", "tokenizer_config.json",
         "vocab.json", "README.md"]
if args[2] == "Qwen/Qwen3-ASR-1.7B":
    files += ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
              "model.safetensors.index.json"]
else:
    assert args[2] == "Qwen/Qwen3-ForcedAligner-0.6B"
    files += ["model.safetensors"]
root = Path(args[args.index("--local-dir") + 1])
root.mkdir(parents=True, exist_ok=True)
for name in files:
    (root / name).write_bytes(os.environ["TEST_HF_CONTENT"].encode())
with Path(os.environ["TEST_HF_LOG"]).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
"""
    )
    executable.chmod(0o755)
    log = tmp_path / "hf-calls.jsonl"
    monkeypatch.setenv("TEST_HF_LOG", str(log))
    monkeypatch.setenv("TEST_HF_CONTENT", content)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    return log


def test_qwen_weights_use_hf_revisions_and_verified_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json
    import re

    script = _download_script(tmp_path)
    content = "verified fixture model"
    checksum = hashlib.sha256(content.encode()).hexdigest()
    script.write_text(re.sub(r"[0-9a-f]{64}", checksum, script.read_text()))
    log = _fake_hf(tmp_path, monkeypatch, content)
    translation = tmp_path / "models/Qwen3.5-9B-AWQ-4bit"
    translation.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ):
        (translation / name).write_text(content)
    vad = tmp_path / "models/silero-vad/silero_vad.onnx"
    vad.parent.mkdir()
    vad.write_text(content)
    result = subprocess.run(
        ["bash", str(script), "--asr-backend", "qwen"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[2] for call in calls] == [
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ForcedAligner-0.6B",
    ]
    assert [call[call.index("--revision") + 1] for call in calls] == [
        "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    ]
    for call in calls:
        assert ".download." in call[call.index("--local-dir") + 1]
        assert call[3] == "--revision"
        model = tmp_path / "models" / call[2].split("/")[1]
        assert (model / "config.json").read_text() == content
        assert all(file.read_text() == content for file in model.glob("*.safetensors"))
    assert not list((tmp_path / "models").glob(".download.*"))
    checked = subprocess.run(
        ["bash", str(script), "--asr-backend", "qwen", "--check"],
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0
    assert len(log.read_text().splitlines()) == 2
