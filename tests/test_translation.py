import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pysrt
import pytest

import src.translation as translation_module
from src.translation import (
    InferenceResult,
    TranslationOptions,
    TranslationTask,
    VLLMTranslator,
)
from src.translation_requests import TranslationRequest
from src.translation_requests import TranslationOutputError, build_translation_requests


class FakeBatchTranslator(VLLMTranslator):
    def __init__(self, model_path: Path, malformed_request: int | None = None) -> None:
        super().__init__(model_path)
        self.malformed_request = malformed_request
        self.inference_calls = 0
        self.release_calls = 0

    def _run_inference(
        self,
        requests: Sequence[TranslationRequest],
        target_lang: str,
    ) -> list[InferenceResult]:
        self.inference_calls += 1
        outputs: list[InferenceResult] = []
        for index, request in enumerate(requests):
            text = "not-json"
            if index != self.malformed_request:
                text = json.dumps(
                    {cue_id: f"translated {cue_id}" for cue_id in request.window_ids}
                )
            outputs.append(InferenceResult(text, "stop"))
        return outputs

    def _release_engine(self) -> None:
        self.release_calls += 1


def _write_srt(path: Path, texts: tuple[str, ...]) -> None:
    items = [
        pysrt.SubRipItem(
            index=index,
            start=pysrt.SubRipTime(milliseconds=index * 1_000),
            end=pysrt.SubRipTime(milliseconds=(index + 1) * 1_000),
            text=text,
        )
        for index, text in enumerate(texts, start=1)
    ]
    pysrt.SubRipFile(items=items).save(str(path), encoding="utf-8")


def test_translate_many_submits_all_movies_once_and_releases_once(
    tmp_path: Path,
) -> None:
    # Given: Two prepared subtitle files and a fake local inference boundary.
    first_input = tmp_path / "first.srt"
    second_input = tmp_path / "second.srt"
    _write_srt(first_input, ("first",))
    _write_srt(second_input, ("second",))
    translator = FakeBatchTranslator(tmp_path)
    tasks = (
        TranslationTask(first_input, tmp_path / "first.zh.srt"),
        TranslationTask(second_input, tmp_path / "second.zh.srt"),
    )

    # When: Both movies are translated as one batch.
    failures = translator.translate_many(
        tasks,
        TranslationOptions(target_lang="Chinese", translated_only=True),
    )

    # Then: One inference submission and one release produce both outputs.
    assert failures == []
    assert translator.inference_calls == 1
    assert translator.release_calls == 1
    assert all(task.output_path.is_file() for task in tasks)


def test_translate_many_isolates_one_movies_malformed_output(tmp_path: Path) -> None:
    # Given: Two movies where only the first model response is malformed.
    first_input = tmp_path / "first.srt"
    second_input = tmp_path / "second.srt"
    _write_srt(first_input, ("first",))
    _write_srt(second_input, ("second",))
    translator = FakeBatchTranslator(tmp_path, malformed_request=0)
    tasks = (
        TranslationTask(first_input, tmp_path / "first.zh.srt"),
        TranslationTask(second_input, tmp_path / "second.zh.srt"),
    )

    # When: Structured outputs are parsed per owning movie.
    failures = translator.translate_many(tasks, TranslationOptions())

    # Then: The invalid movie is withheld while the healthy movie is published.
    assert [failure.task for failure in failures] == [tasks[0]]
    assert not tasks[0].output_path.exists()
    assert tasks[1].output_path.is_file()


def test_translate_many_isolates_one_unreadable_input(tmp_path: Path) -> None:
    # Given: One missing subtitle input and one readable subtitle input.
    missing_input = tmp_path / "missing.srt"
    healthy_input = tmp_path / "healthy.srt"
    _write_srt(healthy_input, ("healthy",))
    translator = FakeBatchTranslator(tmp_path)
    tasks = (
        TranslationTask(missing_input, tmp_path / "missing.zh.srt"),
        TranslationTask(healthy_input, tmp_path / "healthy.zh.srt"),
    )

    # When: The translation batch prepares both movies.
    failures = translator.translate_many(tasks, TranslationOptions())

    # Then: Only the unreadable movie fails and the healthy movie is inferred.
    assert [failure.task for failure in failures] == [tasks[0]]
    assert not tasks[0].output_path.exists()
    assert tasks[1].output_path.is_file()
    assert translator.inference_calls == 1
    assert translator.release_calls == 1


def test_translate_many_isolates_one_output_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Two readable movies where only the first destination cannot be saved.
    first_input = tmp_path / "first.srt"
    second_input = tmp_path / "second.srt"
    _write_srt(first_input, ("first",))
    _write_srt(second_input, ("second",))
    translator = FakeBatchTranslator(tmp_path)
    tasks = (
        TranslationTask(first_input, tmp_path / "first.zh.srt"),
        TranslationTask(second_input, tmp_path / "second.zh.srt"),
    )
    original_save = translation_module._save_subtitles

    def selective_save(path: Path, subtitles: pysrt.SubRipFile) -> None:
        if path == tasks[0].output_path:
            raise OSError("destination is unavailable")
        original_save(path, subtitles)

    monkeypatch.setattr(
        translation_module,
        "_save_subtitles",
        selective_save,
    )

    # When: Both model responses are valid but one publication fails.
    failures = translator.translate_many(tasks, TranslationOptions())

    # Then: The failed destination is isolated and the second movie is published.
    assert [failure.task for failure in failures] == [tasks[0]]
    assert not tasks[0].output_path.exists()
    assert tasks[1].output_path.is_file()
    assert translator.inference_calls == 1
    assert translator.release_calls == 1


def test_run_inference_caps_each_translation_at_2048_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: One translation request and a recorded native vLLM boundary.
    translator = VLLMTranslator(tmp_path)
    request = build_translation_requests(("source",))[0]
    fake_llm = MagicMock()
    fake_llm.chat.return_value = []
    fake_vllm = ModuleType("vllm")
    fake_sampling_params = ModuleType("vllm.sampling_params")

    class FakeStructuredOutputsParams:
        def __init__(self, *, json: str) -> None:
            self.json = json

    class FakeSamplingParams:
        def __init__(
            self,
            *,
            temperature: float,
            top_p: float,
            top_k: int,
            presence_penalty: float,
            max_tokens: int,
            skip_special_tokens: bool,
            structured_outputs: FakeStructuredOutputsParams,
        ) -> None:
            self.max_tokens = max_tokens

    setattr(fake_vllm, "SamplingParams", FakeSamplingParams)
    setattr(
        fake_sampling_params,
        "StructuredOutputsParams",
        FakeStructuredOutputsParams,
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)
    monkeypatch.setattr(translator, "_load_engine", lambda: fake_llm)

    # When: The request is prepared for vLLM inference.
    outputs = translator._run_inference((request,), "Chinese")

    # Then: Its generation budget is capped without changing request batching.
    sampling_params = fake_llm.chat.call_args.kwargs["sampling_params"]
    assert outputs == []
    assert [params.max_tokens for params in sampling_params] == [2048]


def test_length_failure_identifies_the_request_cues_and_generated_size() -> None:
    # Given: A later request whose model output exhausted its generation budget.
    request = build_translation_requests(tuple(f"cue {index}" for index in range(24)))[
        1
    ]
    output = InferenceResult("partial", "length")

    # When / Then: The error identifies both owned and contextual cues.
    with pytest.raises(
        TranslationOutputError,
        match=(r"core cues 17-24 \(window 13-24\).*'length'.*7 characters"),
    ):
        _ = VLLMTranslator._parse_output(request, output)


def test_shared_srt_writer_keeps_existing_output_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pysrt
    from src.utils import save_subtitles_atomic

    output = tmp_path / "result.srt"
    output.write_text("previous result", encoding="utf-8")

    def failed_save(self: pysrt.SubRipFile, path: str, encoding: str) -> None:
        Path(path).write_text("partial write", encoding=encoding)
        raise OSError("write interrupted")

    monkeypatch.setattr(pysrt.SubRipFile, "save", failed_save)
    with pytest.raises(OSError, match="write interrupted"):
        save_subtitles_atomic(output, pysrt.SubRipFile())
    assert output.read_text() == "previous result"
    assert not list(tmp_path.glob(".result.srt.*.tmp"))


@pytest.mark.parametrize("backend", ("translation", "asr", "align"))
@pytest.mark.parametrize("failure", (False, True))
def test_native_logging_keeps_warnings_and_errors_without_initialization_noise(
    tmp_path: Path, backend: str, failure: bool
) -> None:
    import os
    import subprocess

    # Given: The installed vLLM logger with only model construction substituted.
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
import vllm

def load_model(**kwargs):
    from vllm.logger import init_logger
    native = init_logger('vllm.logging_check')
    native.debug('native-debug-sentinel')
    native.info('native-info-sentinel')
    native.warning('native-warning-sentinel')
    if kwargs.get('use_tqdm_on_load', True):
        print('weight-progress-sentinel', file=sys.stderr)
    if sys.argv[3] == '1':
        try:
            raise RuntimeError('native-failure-detail')
        except RuntimeError:
            native.exception('native-error-sentinel')
            raise
    return SimpleNamespace()

vllm.LLM = load_model
if sys.argv[1] == 'translation':
    from src.translation import VLLMTranslator
    VLLMTranslator(Path(sys.argv[2]))._load_engine()
else:
    from src.transcribe import QwenWorker
    QwenWorker._load_model(sys.argv[1])
"""
    environment = os.environ.copy()
    environment.pop("VLLM_LOGGING_LEVEL", None)
    environment.pop("VLLM_LOGGING_STREAM", None)

    # When: Each production entry point initializes its runtime in a fresh process.
    completed = subprocess.run(
        [sys.executable, "-c", script, backend, str(tmp_path), str(int(failure))],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )

    # Then: Warning/error details survive; initialization noise stays hidden.
    output = completed.stdout + completed.stderr
    assert "native-info-sentinel" not in output
    assert "native-debug-sentinel" not in output
    assert "weight-progress-sentinel" not in output
    assert "native-warning-sentinel" in completed.stderr
    assert (completed.returncode != 0) is failure
    if failure:
        assert "native-error-sentinel" in completed.stderr
        assert "RuntimeError: native-failure-detail" in completed.stderr
        assert "Traceback" in completed.stderr
