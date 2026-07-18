import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, final
from unittest.mock import Mock

import pytest
from tqdm import tqdm

import src.transcribe as transcribe_module
from src.transcribe import Transcriber

FAKE_WHISPER: Final = """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
Path(os.environ["FAKE_RECORD"]).write_text("\\n".join(arguments), encoding="utf-8")
Path(os.environ["FAKE_PID"]).write_text(str(os.getpid()), encoding="utf-8")
scenario = os.environ.get("FAKE_SCENARIO", "success")
prefix = arguments[arguments.index("--output-file") + 1]

if scenario == "diagnostics":
    for index in range(80):
        print(f"diagnostic {index}:" + "x" * 100, file=sys.stderr, flush=True)
    raise SystemExit(7)

for line in os.environ.get("FAKE_PROGRESS", "").split("|"):
    if line:
        print(line, file=sys.stderr, flush=True)

if scenario == "linger":
    while True:
        time.sleep(60)
if scenario == "failure":
    raise SystemExit(7)
if scenario == "success":
    Path(f"{prefix}.srt").write_text(os.environ.get("FAKE_SRT", ""), encoding="utf-8")
"""


@dataclass(frozen=True, slots=True)
class FakeRuntime:
    cli: Path
    model: Path
    vad_model: Path
    audio: Path


@final
class RecordingProgress:
    instances: list["RecordingProgress"] = []

    def __init__(
        self, *, total: int, desc: str, unit: str, disable: bool, ncols: int, nrows: int
    ) -> None:
        self.disable = disable
        self.updates: list[int] = []
        self.instances.append(self)

    def __enter__(self) -> "RecordingProgress":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, value: int) -> None:
        self.updates.append(value)


@pytest.fixture
def fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FakeRuntime:
    cli = tmp_path / "whisper-cli"
    _ = cli.write_text(FAKE_WHISPER, encoding="utf-8")
    _ = cli.chmod(0o755)
    model = tmp_path / "ggml-large-v3-turbo.bin"
    vad_model = tmp_path / "ggml-silero-v6.2.0.bin"
    audio = tmp_path / "movie.part.wav"
    for path in (model, vad_model, audio):
        _ = path.write_bytes(b"fixture")
    monkeypatch.setattr(transcribe_module, "WHISPER_CLI_PATH", cli)
    monkeypatch.setattr(transcribe_module, "WHISPER_MODEL_PATH", model)
    monkeypatch.setattr(transcribe_module, "WHISPER_VAD_MODEL_PATH", vad_model)
    monkeypatch.setenv("FAKE_RECORD", str(tmp_path / "argv.txt"))
    monkeypatch.setenv("FAKE_PID", str(tmp_path / "pid.txt"))
    return FakeRuntime(cli, model, vad_model, audio)


def _recorded_arguments(runtime: FakeRuntime) -> list[str]:
    return (runtime.cli.parent / "argv.txt").read_text(encoding="utf-8").splitlines()


def _assert_process_is_gone(runtime: FakeRuntime) -> None:
    pid = int((runtime.cli.parent / "pid.txt").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_transcribe_uses_the_fixed_profile_and_auto_language(
    fake_runtime: FakeRuntime,
    tmp_path: Path,
) -> None:
    # Given: A managed native runtime and an audio file with a dotted stem.
    output_dir = tmp_path / "subtitles"

    # When: Transcription runs without an explicit language.
    result = Transcriber().transcribe(fake_runtime.audio, output_dir, quiet=True)

    # Then: The fixed profile reaches the native CLI and preserves the dotted stem.
    arguments = _recorded_arguments(fake_runtime)
    assert result == output_dir.resolve() / "movie.part.srt"
    assert result.read_bytes() == b""
    assert arguments[:6] == [
        "--model",
        str(fake_runtime.model),
        "--vad-model",
        str(fake_runtime.vad_model),
        "--file",
        str(fake_runtime.audio.resolve()),
    ]
    assert arguments[6:8] == ["--output-file", arguments[7]]
    assert Path(arguments[7]).parent.parent == output_dir.resolve()
    assert Path(arguments[7]).name == fake_runtime.audio.stem
    assert arguments[8:] == [
        "--output-srt",
        "--language",
        "auto",
        "--processors",
        "1",
        "--beam-size",
        "5",
        "--max-context",
        "48",
        "--vad",
        "--vad-threshold",
        "0.15",
        "--vad-min-speech-duration-ms",
        "250",
        "--vad-min-silence-duration-ms",
        "120",
        "--vad-max-speech-duration-s",
        "30",
        "--vad-speech-pad-ms",
        "250",
        "--vad-samples-overlap",
        "0",
        "--no-prints",
        "--print-progress",
    ]


def test_transcribe_passes_an_explicit_language_and_overwrites_on_success(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: An existing subtitle and a new native result.
    destination = fake_runtime.audio.with_suffix(".srt")
    _ = destination.write_text("old", encoding="utf-8")
    monkeypatch.setenv("FAKE_SRT", "new")

    # When: The caller supplies an explicit language code.
    result = Transcriber().transcribe(fake_runtime.audio, language="fr", quiet=True)

    # Then: The successful result replaces the old file and the language is forwarded.
    arguments = _recorded_arguments(fake_runtime)
    assert result.read_text(encoding="utf-8") == "new"
    assert arguments[arguments.index("--language") + 1] == "fr"


@pytest.mark.parametrize("quiet", (False, True))
def test_transcribe_reports_only_monotonic_valid_progress(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
    quiet: bool,
) -> None:
    # Given: Native progress containing noise, overflow, and a regression.
    RecordingProgress.instances.clear()
    monkeypatch.setattr(transcribe_module, "tqdm", RecordingProgress)
    monkeypatch.setenv(
        "FAKE_PROGRESS",
        "|".join(
            (
                "whisper_print_progress_callback: progress = 10%",
                "garbage",
                "whisper_print_progress_callback: progress = 101%",
                "whisper_print_progress_callback: progress = 60%",
                "whisper_print_progress_callback: progress = 20%",
                "whisper_print_progress_callback: progress = 100%",
            )
        ),
    )

    # When: The native process emits stderr while transcription runs.
    _ = Transcriber().transcribe(fake_runtime.audio, quiet=quiet)

    # Then: Display advances monotonically while quiet changes only its visibility.
    assert RecordingProgress.instances[-1].updates == [10, 50, 40]
    assert RecordingProgress.instances[-1].disable is quiet


def test_transcribe_progress_is_visible_in_a_zero_width_pty(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A real PTY with no reported width and a native completion update.
    progress_line = "whisper_print_progress_callback: progress = 100%"
    monkeypatch.setenv("FAKE_PROGRESS", progress_line)
    zero_size = os.terminal_size((0, 0))
    progress_factory = Mock(wraps=tqdm)
    master, slave = os.openpty()

    # When: The adapter renders progress to that terminal.
    try:
        with (
            os.fdopen(slave, "w", encoding="utf-8") as terminal,
            monkeypatch.context() as context,
        ):
            context.setattr(sys, "stderr", terminal)
            context.setattr("shutil.get_terminal_size", Mock(return_value=zero_size))
            context.setattr("src.transcribe.tqdm", progress_factory)
            _ = Transcriber().transcribe(fake_runtime.audio)
        rendered = os.read(master, 4096).decode(errors="replace")
    finally:
        os.close(master)

    # Then: The user can observe completion despite the missing window metadata.
    assert all(token in rendered for token in ("Transcribing", "100%"))
    options = progress_factory.call_args.kwargs
    assert options["ncols"] == 80 and options["nrows"] == 24


@pytest.mark.parametrize("scenario", ("failure", "no-output"))
def test_transcribe_preserves_an_existing_subtitle_on_native_failure(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    # Given: An existing subtitle and a native failure mode.
    destination = fake_runtime.audio.with_suffix(".srt")
    _ = destination.write_bytes(b"trusted-old-subtitle")
    monkeypatch.setenv("FAKE_SCENARIO", scenario)

    # When / Then: The failed attempt raises and leaves the destination byte-identical.
    with pytest.raises(RuntimeError):
        _ = Transcriber().transcribe(fake_runtime.audio, quiet=True)
    assert destination.read_bytes() == b"trusted-old-subtitle"
    _assert_process_is_gone(fake_runtime)


def test_transcribe_bounds_native_diagnostics(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A failing native process that floods stderr.
    monkeypatch.setenv("FAKE_SCENARIO", "diagnostics")

    # When: The adapter reports the native failure.
    with pytest.raises(RuntimeError) as captured:
        _ = Transcriber().transcribe(fake_runtime.audio, quiet=True)

    # Then: The useful tail is retained without returning the entire stream.
    message = str(captured.value)
    assert "exit code 7" in message
    assert "diagnostic 79:" in message
    assert "diagnostic 0:" not in message
    assert len(message) < 4_000


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_transcribe_terminates_the_child_on_python_failure_or_cancellation(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[RuntimeError] | type[KeyboardInterrupt],
) -> None:
    # Given: A lingering child, an existing subtitle, and a Python-side interruption.
    destination = fake_runtime.audio.with_suffix(".srt")
    _ = destination.write_bytes(b"trusted-old-subtitle")
    monkeypatch.setenv("FAKE_SCENARIO", "linger")
    monkeypatch.setenv(
        "FAKE_PROGRESS", "whisper_print_progress_callback: progress = 10%"
    )

    def raise_failure(_line: str) -> int | None:
        raise failure_type("forced parser interruption")

    monkeypatch.setattr(transcribe_module, "parse_progress", raise_failure)

    # When / Then: Propagation happens only after preserving output and reaping the child.
    with pytest.raises(failure_type):
        _ = Transcriber().transcribe(fake_runtime.audio, quiet=True)
    assert destination.read_bytes() == b"trusted-old-subtitle"
    _assert_process_is_gone(fake_runtime)


@pytest.mark.parametrize(
    "constant_name",
    ("WHISPER_CLI_PATH", "WHISPER_MODEL_PATH", "WHISPER_VAD_MODEL_PATH"),
)
def test_transcribe_names_a_missing_managed_prerequisite(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    constant_name: str,
) -> None:
    # Given: One exact managed runtime prerequisite is missing.
    missing = (tmp_path / f"missing-{constant_name}").resolve()
    monkeypatch.setattr(transcribe_module, constant_name, missing)

    # When / Then: Validation names the unavailable absolute path.
    with pytest.raises(FileNotFoundError, match=str(missing)):
        _ = Transcriber().transcribe(fake_runtime.audio, quiet=True)
