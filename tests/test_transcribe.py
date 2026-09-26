import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace, TracebackType
from typing import Final, final
from unittest.mock import Mock

import pytest
from tqdm import tqdm

import src.transcribe as transcribe_module
import src.whisper_batch as whisper_batch_module
from src.transcribe import Transcriber, TranscriptionTask

FAKE_WHISPER: Final = """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
Path(os.environ["FAKE_RECORD"]).write_text("\\n".join(arguments), encoding="utf-8")
Path(os.environ["FAKE_PID"]).write_text(str(os.getpid()), encoding="utf-8")
scenario = os.environ.get("FAKE_SCENARIO", "success")
audio_paths = arguments[arguments.index("--print-progress") + 1:]

if scenario == "diagnostics":
    for index in range(80):
        print(f"diagnostic {index}:" + "x" * 100, file=sys.stderr, flush=True)
    raise SystemExit(7)

for index, audio_path in enumerate(audio_paths):
    print(f"read_audio_data: reading audio data from '{audio_path}' ...", file=sys.stderr, flush=True)
    for line in os.environ.get("FAKE_PROGRESS", "").split("|"):
        if line:
            print(line, file=sys.stderr, flush=True)
    if scenario == "linger":
        while True:
            time.sleep(60)
    if scenario == "failure":
        raise SystemExit(7)
    if scenario == "success" and index != int(os.environ.get("FAKE_SKIP_INDEX", "-1")):
        Path(f"{audio_path}.srt").write_text(os.environ.get("FAKE_SRT", ""), encoding="utf-8")
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


def test_transcribe_can_disable_vad_and_preserves_a_dotted_stem(
    fake_runtime: FakeRuntime,
    tmp_path: Path,
) -> None:
    # Given: A managed runtime, a dotted audio stem, and explicitly disabled VAD.
    output_dir = tmp_path / "subtitles"

    # When: Transcription runs without an explicit language or VAD.
    result = Transcriber().transcribe(
        fake_runtime.audio,
        output_dir,
        quiet=True,
        enable_vad=False,
    )

    # Then: The fixed profile reaches the native CLI and preserves the dotted stem.
    arguments = _recorded_arguments(fake_runtime)
    assert result == output_dir.resolve() / "movie.part.srt"
    assert result.read_bytes() == b""
    assert arguments[:2] == [
        "--model",
        str(fake_runtime.model),
    ]
    assert arguments[2:-1] == [
        "--output-srt",
        "--language",
        "auto",
        "--processors",
        "1",
        "--beam-size",
        "5",
        "--max-context",
        "0",
        "--suppress-nst",
        "--no-prints",
        "--print-progress",
    ]
    assert Path(arguments[-1]).parent.parent == output_dir.resolve()


def test_transcribe_uses_the_best_tested_vad_profile_by_default(
    fake_runtime: FakeRuntime,
) -> None:
    # Given / When: The caller uses the default integrated VAD profile.
    _ = Transcriber().transcribe(fake_runtime.audio, quiet=True)

    # Then: The VAD model and the best parameters from the three-sample study apply.
    arguments = _recorded_arguments(fake_runtime)
    assert arguments[arguments.index("--max-context") + 1] == "48"
    assert arguments[arguments.index("--vad-model") + 1] == str(fake_runtime.vad_model)
    assert arguments[arguments.index("--vad-threshold") + 1] == "0.01"
    assert arguments[arguments.index("--vad-min-speech-duration-ms") + 1] == "100"
    assert arguments[arguments.index("--vad-min-silence-duration-ms") + 1] == "120"
    assert arguments[arguments.index("--vad-max-speech-duration-s") + 1] == "10"
    assert arguments[arguments.index("--vad-speech-pad-ms") + 1] == "500"
    assert arguments[arguments.index("--vad-samples-overlap") + 1] == "0"
    assert "--suppress-nst" not in arguments


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
    monkeypatch.setattr(whisper_batch_module, "tqdm", RecordingProgress)
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
            context.setattr("src.whisper_batch.tqdm", progress_factory)
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

    monkeypatch.setattr(whisper_batch_module, "parse_progress", raise_failure)

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


def test_transcribe_does_not_require_the_vad_model_when_disabled(
    fake_runtime: FakeRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: The optional managed VAD model is unavailable.
    monkeypatch.setattr(
        transcribe_module,
        "WHISPER_VAD_MODEL_PATH",
        tmp_path / "missing-vad-model.bin",
    )

    # When / Then: Explicitly disabled VAD does not require its managed model.
    assert (
        Transcriber()
        .transcribe(
            fake_runtime.audio,
            quiet=True,
            enable_vad=False,
        )
        .is_file()
    )


def test_transcribe_many_uses_one_process_and_publishes_each_output(
    fake_runtime: FakeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Two audio tasks sharing one managed native runtime.
    second_audio = tmp_path / "second.wav"
    second_audio.write_bytes(b"fixture")
    first_output = tmp_path / "first.srt"
    second_output = tmp_path / "second.srt"
    monkeypatch.setenv("FAKE_SRT", "published")

    # When: Both files are submitted together.
    failures = Transcriber().transcribe_many(
        (
            TranscriptionTask(fake_runtime.audio, first_output),
            TranscriptionTask(second_audio, second_output),
        ),
        language="en",
        quiet=True,
    )

    # Then: One invocation publishes both destinations atomically.
    arguments = _recorded_arguments(fake_runtime)
    assert failures == []
    assert arguments.count("--model") == 1
    assert len(arguments[arguments.index("--print-progress") + 1 :]) == 2
    assert first_output.read_text(encoding="utf-8") == "published"
    assert second_output.read_text(encoding="utf-8") == "published"


def test_transcribe_many_isolates_a_missing_native_output(
    fake_runtime: FakeRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: A native batch that omits only its second output.
    second_audio = tmp_path / "second.wav"
    second_audio.write_bytes(b"fixture")
    first_output = tmp_path / "first.srt"
    second_output = tmp_path / "second.srt"
    monkeypatch.setenv("FAKE_SRT", "published")
    monkeypatch.setenv("FAKE_SKIP_INDEX", "1")
    tasks = (
        TranscriptionTask(fake_runtime.audio, first_output),
        TranscriptionTask(second_audio, second_output),
    )

    # When: The native process exits after the partial batch.
    failures = Transcriber().transcribe_many(tasks, quiet=True)

    # Then: The complete first result is retained and only the second fails.
    assert [failure.task for failure in failures] == [tasks[1]]
    assert first_output.is_file()
    assert not second_output.exists()


def test_source_language_is_optional_and_parsed_from_model_output() -> None:
    from src.transcribe import QwenASR

    assert QwenASR.resolve_language(None) is None
    assert QwenASR.resolve_language("fr") == "French"
    result = QwenASR.parse_output("language French<asr_text>Bonjour. Merci !", None)
    assert result == ("French", "Bonjour. Merci !")
    forced = QwenASR.parse_output("Hello world.", "English")
    assert forced == ("English", "Hello world.")


@pytest.fixture
def qwen_llm(monkeypatch: pytest.MonkeyPatch) -> Mock:
    fake_vllm = ModuleType("vllm")
    fake_inputs = ModuleType("vllm.inputs")
    setattr(fake_vllm, "SamplingParams", SimpleNamespace)
    setattr(fake_inputs, "TextPrompt", dict)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_inputs)
    return Mock()


def _qwen_output(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(text=text, finish_reason=finish_reason)]
    )


@pytest.mark.parametrize(
    "text,finish_reason",
    [("Partial", "length"), ("Echo!" * 25, "stop"), ("a" * 21, "stop")],
)
@pytest.mark.parametrize("show_progress", (False, True))
def test_qwen_retries_only_suspect_windows_once(
    qwen_llm: Mock, text: str, finish_reason: str, show_progress: bool
) -> None:
    from io import StringIO
    import numpy as np
    from src.aligner import AlignmentRecord
    from src.transcribe import QwenASR

    records = [
        AlignmentRecord(
            segment_id="a", start=0, end=16000, text="", language="English"
        ),
        AlignmentRecord(
            segment_id="b", start=32000, end=64000, text="", language="French"
        ),
        AlignmentRecord(
            segment_id="c", start=64000, end=80000, text="", language="English"
        ),
    ]
    audio = np.arange(80000, dtype=np.float32)
    qwen_llm.generate.side_effect = [
        [
            _qwen_output("Keep this."),
            _qwen_output(text, finish_reason),
            _qwen_output("Last."),
        ],
        [_qwen_output("Bonjour.")],
    ]
    stream = StringIO() if show_progress else None
    results = QwenASR(qwen_llm, stream).transcribe(records, audio)
    assert [r.text for r in results] == ["Keep this.", "Bonjour.", "Last."]
    assert [(r.segment_id, r.start, r.end, r.language) for r in results] == [
        (r.segment_id, r.start, r.end, r.language) for r in records
    ]
    assert all(r.error is None for r in results)
    first, retry = qwen_llm.generate.call_args_list
    assert len(first.args[0]) == 3 and len(retry.args[0]) == 1
    assert retry.args[0][0]["prompt"] == first.args[0][1]["prompt"]
    np.testing.assert_array_equal(
        retry.args[0][0]["multi_modal_data"]["audio"][0], audio[32000:64000]
    )
    assert [call.args[1].repetition_penalty for call in (first, retry)] == [1.0, 1.1]
    assert all(
        call.args[1].temperature == 0 and call.args[1].max_tokens == 4096
        for call in (first, retry)
    )
    for call in (first, retry):
        callback = call.kwargs["use_tqdm"]
        if show_progress:
            with callback(total=len(call.args[0]), desc="Processed prompts") as bar:
                bar.update(len(call.args[0]))
                bar.refresh()
        else:
            assert callback is False
    if stream is not None:
        assert "Qwen ASR:" in stream.getvalue()
        assert "Qwen ASR retry:" in stream.getvalue()


def test_qwen_preserves_normal_repetition_without_retry(qwen_llm: Mock) -> None:
    import numpy as np
    from src.aligner import AlignmentRecord
    from src.transcribe import QwenASR

    record = AlignmentRecord(segment_id="a", start=0, end=16000, text="")
    text = "Yes, yes, yes!"
    qwen_llm.generate.return_value = [_qwen_output("language English<asr_text>" + text)]
    result = QwenASR(qwen_llm).transcribe([record], np.zeros(16000, dtype=np.float32))
    assert result[0].text == text and result[0].language == "English"
    assert qwen_llm.generate.call_count == 1


@pytest.mark.parametrize(
    "text,reason", [("Partial", "length"), ("Again!" * 25, "stop")]
)
def test_qwen_unrecovered_window_fails_in_asr_with_time_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qwen_llm: Mock,
    text: str,
    reason: str,
) -> None:
    import numpy as np
    from src.aligner import AlignmentRecord, write_records
    from src.transcribe import QwenWorker

    qwen_llm.generate.return_value = [_qwen_output(text, reason)]
    write_records(
        tmp_path / "prepare.json",
        [
            AlignmentRecord(
                segment_id="w1", start=16000, end=32000, text="", language="English"
            )
        ],
    )
    monkeypatch.setattr(QwenWorker, "_load_model", staticmethod(lambda stage: qwen_llm))
    monkeypatch.setattr(
        transcribe_module, "read_audio", lambda path: np.zeros(32000, dtype=np.float32)
    )
    QwenWorker("asr", "English", True).run([(tmp_path / "audio.wav", tmp_path)])
    assert qwen_llm.generate.call_count == 2
    assert not (tmp_path / "asr.json").exists()
    error = (tmp_path / "asr.error").read_text()
    assert "w1" in error and "1.000-2.000 s" in error and "after one retry" in error


@pytest.mark.parametrize("quiet", (False, True))
@pytest.mark.parametrize(
    "failure", (None, RuntimeError("worker failed"), KeyboardInterrupt())
)
def test_qwen_batch_always_removes_temporary_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    quiet: bool,
    failure: BaseException | None,
) -> None:
    import logging
    from src.aligner import AlignmentRecord, write_records
    from src.transcribe import QwenBatchRunner

    audio = tmp_path / "audio.wav"
    audio.touch()
    task = TranscriptionTask(audio, tmp_path / "output.srt")
    monkeypatch.setattr("src.transcribe.tempfile.tempdir", str(tmp_path))

    def fake_stage(pending, stage, *args, quiet: bool):
        if failure is not None:
            raise failure
        for directory in pending.values():
            write_records(
                directory / f"{stage}.json",
                [AlignmentRecord(segment_id="a", start=0, end=16000, text="Hello.")],
            )
        return pending, []

    monkeypatch.setattr(QwenBatchRunner, "_validate_models", lambda *args: None)
    monkeypatch.setattr(
        QwenBatchRunner,
        "_run_stage_process",
        staticmethod(fake_stage),
    )
    with caplog.at_level(logging.INFO, logger="SimpleRadeonSubs"):
        if failure is None:
            assert QwenBatchRunner().run((task,), "en", quiet, True) == []
        else:
            with pytest.raises(type(failure)):
                QwenBatchRunner().run((task,), "en", quiet, True)
    expected = {audio}
    if failure is None:
        expected.add(task.output_path)
        assert "Hello." in task.output_path.read_text()
    assert set(tmp_path.iterdir()) == expected
    assert "Qwen diagnostics for" not in caplog.text
    assert ("Qwen prepare: 1 file(s)" in caplog.text) is not quiet


def test_qwen_batch_reuses_models_and_isolates_failed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import src.aligner as aligner_module
    import src.transcribe as transcribe_module
    from src.aligner import AlignmentRecord, read_records, write_records

    directories = [tmp_path / name for name in ("broken", "first", "second")]
    for directory in directories:
        directory.mkdir()
    for directory in directories[1:]:
        write_records(
            directory / "prepare.json",
            [
                AlignmentRecord(segment_id="short", start=0, end=32000, text=""),
                AlignmentRecord(segment_id="long", start=64000, end=128000, text=""),
            ],
        )
    loads: list[str] = []
    requests: list[tuple[str, list[str]]] = []

    class FakeASR:
        def __init__(self, llm, progress=None) -> None:
            loads.append("asr")

        def transcribe(
            self, records: list[AlignmentRecord], audio
        ) -> list[AlignmentRecord]:
            requests.append(("asr", [r.segment_id for r in records]))
            return [r.model_copy(update={"text": "Hello."}) for r in records]

    class FakeAligner:
        def __init__(self, llm, progress=None) -> None:
            loads.append("align")

        def align(self, records: list[AlignmentRecord], audio) -> list[AlignmentRecord]:
            requests.append(("align", [r.segment_id for r in records]))
            return records

    monkeypatch.setattr(transcribe_module, "QwenASR", FakeASR)
    monkeypatch.setattr(aligner_module, "ForcedAligner", FakeAligner)
    monkeypatch.setattr(
        transcribe_module.QwenWorker, "_load_model", staticmethod(lambda stage: None)
    )
    monkeypatch.setattr(
        transcribe_module, "read_audio", lambda path: np.zeros(128000, dtype=np.float32)
    )
    inputs = [(directory / "audio.wav", directory) for directory in directories]
    transcribe_module.QwenWorker("asr", "English", True).run(inputs)
    transcribe_module.QwenWorker("align", "English", True).run(inputs[1:])
    assert loads == ["asr", "align"]
    assert requests == [
        ("asr", ["short", "long"]),
        ("asr", ["short", "long"]),
        ("align", ["long"]),
        ("align", ["long"]),
    ]
    assert "prepare.json" in (directories[0] / "asr.error").read_text()
    assert read_records(directories[1] / "align.json")[0].units == ()
    assert not list(tmp_path.rglob("*-report.json"))


def test_qwen_short_windows_never_initialize_aligner(tmp_path: Path) -> None:
    from src.aligner import AlignmentRecord, read_records, write_records
    from src.transcribe import QwenBatchRunner

    source = AlignmentRecord(segment_id="short", start=0, end=16000, text="Hello.")
    write_records(
        tmp_path / "asr.json",
        [source],
    )
    task = TranscriptionTask(tmp_path / "absent.wav", tmp_path / "output.srt")
    pending = {task: tmp_path}
    succeeded, failures = QwenBatchRunner._run_stage_process(
        pending, "align", None, True
    )
    assert succeeded == pending and failures == []
    records = read_records(tmp_path / "align.json")
    assert records == [source]
    assert {p.name for p in tmp_path.iterdir()} == {"asr.json", "align.json"}


@pytest.mark.parametrize("quiet", (False, True))
@pytest.mark.parametrize("columns", (0, 40, 120))
def test_qwen_progress_reaches_terminal_before_worker_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet: bool, columns: int,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import select
    import subprocess
    import termios
    from time import monotonic, sleep
    from src.transcribe import QwenBatchRunner

    child = """
import json, logging, os, sys, time
from contextlib import nullcontext
from pathlib import Path
from src.utils import inference_progress

directory = Path(json.loads(sys.stdin.buffer.read())[0][1])
fd = int(sys.argv[4])
print('native-log-sentinel', flush=True)
logging.basicConfig(level=logging.WARNING)
logging.warning('native-warning-sentinel')
with os.fdopen(fd, 'w') if fd >= 0 else nullcontext(None) as terminal:
    factory = inference_progress(terminal, 'Qwen ASR')
    bar = factory(total=3, desc='Processed prompts', dynamic_ncols=True) if factory else None
    if bar is not None:
        bar.update(1)
        bar.refresh()
    (directory / 'ready').touch()
    deadline = time.monotonic() + 10
    while not (directory / 'continue').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('progress test release timed out')
        time.sleep(0.01)
    if bar is not None:
        bar.update(2)
        bar.refresh()
        bar.close()
(directory / 'asr.json').write_text('[]')
"""
    native_run = subprocess.run

    def run_child(command, **kwargs):
        command[2] = child
        if kwargs.get("stderr") is None:
            kwargs["stderr"] = slave
        return native_run(command, **kwargs)

    monkeypatch.setattr(transcribe_module.subprocess, "run", run_child)
    master, slave = os.openpty()
    termios.tcsetwinsize(slave, (24, columns))
    duplicate = os.dup
    monkeypatch.setattr(transcribe_module.os, "dup", lambda fd: duplicate(slave))
    task = TranscriptionTask(tmp_path / "audio.wav", tmp_path / "output.srt")
    pending = {task: tmp_path}
    captured = b""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                QwenBatchRunner._run_stage_process,
                pending,
                "asr",
                None,
                True,
                quiet=quiet,
            )
            try:
                deadline = monotonic() + 10
                while not (tmp_path / "ready").exists() and monotonic() < deadline:
                    sleep(0.01)
                assert (tmp_path / "ready").exists()
                assert not future.done()
                if select.select([master], [], [], 0.1)[0]:
                    captured += os.read(master, 65536)
                assert b"native-warning-sentinel" in captured
                assert (b"1/3" in captured) is not quiet
            finally:
                (tmp_path / "continue").touch()
            assert future.result(timeout=10) == (pending, [])
            while select.select([master], [], [], 0)[0]:
                captured += os.read(master, 65536)
    finally:
        os.close(master)
        os.close(slave)
    assert b"native-log-sentinel" not in captured
    if quiet:
        assert b"Qwen ASR" not in captured
    else:
        assert b"100%" in captured and b"3/3" in captured


@pytest.mark.parametrize("caller", ("pipeline", "cli"))
@pytest.mark.parametrize("unexpected", (False, True))
def test_qwen_file_failures_are_reported_once_by_the_public_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    caller: str,
    unexpected: bool,
) -> None:
    import subprocess
    from src.pipeline import PipelineOptions, PipelinePlan, build_jobs, run_pipeline
    from src.transcribe import QwenBatchRunner

    # Given: Real stage workers fail on multiple inputs without loading models.
    sources = tuple(tmp_path / f"file-{index}.wav" for index in range(2))
    for source in sources:
        source.touch()
    native_run = subprocess.run

    def failing_worker(command: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        command[2] = """
from src.transcribe import QwenWorker, TranscriptionError

def fail(self, audio_path):
    marker = audio_path.stem if audio_path.stem != 'audio' else audio_path.parent.name
    detail = f'{marker}-failure: w00001 (1.000-2.000 s): asr_truncated'
    if UNEXPECTED:
        try:
            raise OSError('native-cause')
        except OSError as error:
            raise ValueError(detail) from error
    raise TranscriptionError(detail)

QwenWorker._prepare = fail
QwenWorker.main()
""".replace("UNEXPECTED", str(unexpected))
        return native_run(command, **kwargs)

    monkeypatch.setattr(transcribe_module.subprocess, "run", failing_worker)
    monkeypatch.setattr(QwenBatchRunner, "_validate_models", lambda *args: None)

    # When: The actual pipeline or standalone CLI handles each returned failure.
    if caller == "pipeline":
        options = PipelineOptions(asr_backend="qwen")
        jobs = build_jobs(PipelinePlan(sources, tmp_path, None, options))
        for job in jobs:
            job.audio_path.parent.mkdir()
            job.audio_path.touch()
        result = run_pipeline(jobs, options)
        assert len(result.failures) == 2
    else:
        for source in sources:
            monkeypatch.setattr(sys, "argv", ["transcribe", str(source), "--asr-backend", "qwen"])
            assert transcribe_module.main() == 1

    # Then: File-specific details appear once, with tracebacks only for unexpected errors.
    messages = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
    assert len(messages) == 2
    for index, message in enumerate(messages):
        assert f"file-{index}" in message
        assert message.count("asr_truncated") == 1
        assert "w00001 (1.000-2.000 s)" in message
        assert ("Traceback" in message) is unexpected
        assert ("native-cause" in message) is unexpected
        assert "Worker output tail" not in message
        assert "exit 0" not in message


@pytest.mark.parametrize("returncode", (0, 9))
@pytest.mark.parametrize("all_completed", (False, True))
def test_qwen_worker_crash_preserves_completed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    returncode: int,
    all_completed: bool,
) -> None:
    import subprocess
    from src.transcribe import QwenBatchRunner, TranscriptionTask

    pending = {}
    for name in ("completed", "reported", "crashed", "not-started"):
        directory = tmp_path / name
        directory.mkdir()
        pending[TranscriptionTask(directory / "audio.wav", directory / "raw.srt")] = (
            directory
        )

    native_run = subprocess.run

    def failing_worker(command: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        command[2] = """
import json, sys
from pathlib import Path

parsed = json.loads(sys.stdin.buffer.read())
if ALL_COMPLETED:
    for audio, directory in parsed:
        Path(directory, 'asr.json').write_text('[]')
else:
    Path(parsed[0][1], 'asr.json').write_text('[]')
    Path(parsed[1][1], 'asr.json').write_text('partial output')
    Path(parsed[1][1], 'asr.error').write_text('decode failed')
print('early detail\\n' + 'x' * 10000 + '\\nfatal worker error', flush=True)
print('native-stderr-failure', file=sys.stderr, flush=True)
sys.exit(RETURN_CODE)
""".replace("ALL_COMPLETED", str(all_completed)).replace("RETURN_CODE", str(returncode))
        return native_run(command, **kwargs)

    monkeypatch.setattr("src.transcribe.subprocess.run", failing_worker)
    succeeded, failures = QwenBatchRunner._run_stage_process(
        pending, "asr", None, True
    )
    tasks = list(pending)
    assert list(succeeded) == (tasks if all_completed else tasks[:1])
    assert [failure.task for failure in failures] == ([] if all_completed else tasks[1:])
    if not all_completed:
        assert "decode failed" in failures[0].reason
        assert all("Worker did not produce records" in failure.reason for failure in failures[1:])
    assert all("Worker output tail" not in failure.reason for failure in failures)
    errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
    if returncode or not all_completed:
        assert len(errors) == 1
        assert errors[0].count(f"exit {returncode}") == 1
        assert errors[0].count("fatal worker error") == 1
        assert "early detail" not in errors[0]
        assert "decode failed" not in errors[0]
    else:
        assert errors == []
    assert "native-stderr-failure" in capfd.readouterr().err
    assert not list(tmp_path.rglob("*.log"))


def test_qwen_publish_preserves_existing_srt_on_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import pysrt
    from src.aligner import AlignmentRecord, write_records
    from src.transcribe import TranscriptionError, TranscriptionTask, QwenBatchRunner

    destination = tmp_path / "output.srt"
    task = TranscriptionTask(tmp_path / "audio.wav", destination)
    destination.write_text("previous result", encoding="utf-8")
    failed = AlignmentRecord(
        segment_id="a", start=0, end=16000, text="Partial", error="asr_truncated"
    )
    write_records(tmp_path / "align.json", [failed])
    with pytest.raises(TranscriptionError, match="asr_truncated"):
        QwenBatchRunner._publish(task, tmp_path)
    assert destination.read_text() == "previous result"
    complete = failed.model_copy(update={"text": "Hello.", "error": None})
    empty = AlignmentRecord(segment_id="b", start=32000, end=64000, text="")
    write_records(tmp_path / "align.json", [complete, empty])
    QwenBatchRunner._publish(task, tmp_path)
    assert pysrt.open(str(destination), encoding="utf-8")[0].text == "Hello."
    assert "1 speech windows returned empty text" in caplog.text
    assert {p.name for p in tmp_path.iterdir()} == {"align.json", "output.srt"}


@pytest.mark.parametrize("unexpected", (False, True))
def test_qwen_publication_failures_return_details_without_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    unexpected: bool,
) -> None:
    from src.aligner import AlignmentRecord, write_records
    from src.transcribe import QwenBatchRunner

    audio = tmp_path / "audio.wav"
    audio.touch()
    task = TranscriptionTask(audio, tmp_path / "output.srt")

    def complete_stage(pending, stage, *args, **kwargs):
        for directory in pending.values():
            if unexpected:
                (directory / "align.json").write_text("invalid JSON")
            else:
                write_records(
                    directory / "align.json",
                    [AlignmentRecord(segment_id="w1", start=0, end=16000, text="x", error="invalid-span")],
                )
        return pending, []

    monkeypatch.setattr(QwenBatchRunner, "_validate_models", lambda *args: None)
    monkeypatch.setattr(QwenBatchRunner, "_run_stage_process", staticmethod(complete_stage))
    failures = QwenBatchRunner().run((task,), None, True, True)
    assert len(failures) == 1 and failures[0].task == task
    assert ("Traceback" in failures[0].reason) is unexpected
    assert ("invalid-span" in failures[0].reason) is not unexpected
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
