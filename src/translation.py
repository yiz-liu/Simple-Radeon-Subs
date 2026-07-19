from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pysrt

from src.config import TRANSLATION_MODEL_PATH
from src.logger import logger
from src.translation_output import _build_final_subtitles, _save_subtitles
from src.translation_requests import (
    TranslationOutputError,
    TranslationRequest,
    TranslationResult,
    build_translation_requests,
    parse_translation_output,
)

if TYPE_CHECKING:
    from vllm import LLM

MODEL_CONTEXT_LENGTH: Final = 4096
MAX_TRANSLATION_TOKENS: Final = 2048


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Normalized text returned from the native vLLM boundary."""

    text: str | None
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class TranslationOptions:
    """Options shared by full-pipeline and standalone translation."""

    target_lang: str = "Chinese"
    translated_only: bool = False


@dataclass(frozen=True, slots=True)
class TranslationTask:
    """One prepared SRT and its final translated destination."""

    input_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class TranslationFailure:
    """A translation failure scoped to one subtitle file."""

    task: TranslationTask
    reason: str


@dataclass(frozen=True, slots=True)
class _PreparedTranslation:
    task: TranslationTask
    subtitles: pysrt.SubRipFile
    requests: tuple[TranslationRequest, ...]


DEFAULT_TRANSLATION_OPTIONS: Final = TranslationOptions()


class VLLMTranslator:
    """Translate subtitle batches with one managed local vLLM engine."""

    def __init__(self, model_path: Path = TRANSLATION_MODEL_PATH) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"vLLM model not found: {self.model_path}")
        self._llm: LLM | None = None

    def translate_many(
        self,
        tasks: Sequence[TranslationTask],
        options: TranslationOptions = DEFAULT_TRANSLATION_OPTIONS,
    ) -> list[TranslationFailure]:
        """Translate multiple SRT files with one inference submission."""
        failures: list[TranslationFailure] = []
        pending: list[_PreparedTranslation] = []
        for task in tasks:
            try:
                item = self._prepare_task(task)
                if item.requests:
                    pending.append(item)
                else:
                    _save_subtitles(item.task.output_path, item.subtitles)
            except (OSError, UnicodeError, pysrt.Error) as error:
                failures.append(TranslationFailure(task, str(error)))

        if not pending:
            return failures

        requests = tuple(request for item in pending for request in item.requests)
        owners = tuple(
            index for index, item in enumerate(pending) for _ in item.requests
        )
        logger.info(
            "Translating %d subtitle file(s) in %d requests via vLLM (model: %s).",
            len(pending),
            len(requests),
            self.model_path.name,
        )
        try:
            outputs = self._run_inference(requests, options.target_lang)
            results: list[list[TranslationResult]] = [[] for _ in pending]
            reasons: dict[int, str] = {}
            for request, output, owner in zip(requests, outputs, owners, strict=True):
                if owner in reasons:
                    continue
                try:
                    results[owner].append(self._parse_output(request, output))
                except TranslationOutputError as error:
                    reasons[owner] = str(error)

            for index, item in enumerate(pending):
                reason = reasons.get(index)
                if reason is not None:
                    failures.append(TranslationFailure(item.task, reason))
                    continue
                try:
                    translations = self._reassemble_subtitles(
                        len(item.subtitles),
                        results[index],
                    )
                    final = _build_final_subtitles(
                        item.subtitles,
                        translations,
                        options.translated_only,
                    )
                    _save_subtitles(item.task.output_path, final)
                except (
                    OSError,
                    UnicodeError,
                    pysrt.Error,
                    TranslationOutputError,
                ) as error:
                    failures.append(TranslationFailure(item.task, str(error)))
            return failures
        finally:
            self._release_engine()

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        options: TranslationOptions = DEFAULT_TRANSLATION_OPTIONS,
    ) -> None:
        """Translate one SRT while preserving every source timestamp."""
        failures = self.translate_many(
            (TranslationTask(input_path, output_path),), options
        )
        if failures:
            raise TranslationOutputError(failures[0].reason)

    def _prepare_task(self, task: TranslationTask) -> _PreparedTranslation:
        subtitles = pysrt.open(str(task.input_path.resolve()), encoding="utf-8")
        requests = tuple(
            build_translation_requests(
                tuple(subtitle.text.replace("\n", " ") for subtitle in subtitles)
            )
        )
        return _PreparedTranslation(task, subtitles, requests)

    def _run_inference(
        self,
        requests: Sequence[TranslationRequest],
        target_lang: str,
    ) -> list[InferenceResult]:
        llm = self._load_engine()
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sampling_params = [
            SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                presence_penalty=1.5,
                max_tokens=MAX_TRANSLATION_TOKENS,
                skip_special_tokens=True,
                structured_outputs=StructuredOutputsParams(
                    json=json.dumps(request.schema, separators=(",", ":"))
                ),
            )
            for request in requests
        ]
        raw_outputs = llm.chat(
            [request.messages(target_lang) for request in requests],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        return [
            InferenceResult(
                text=output.outputs[0].text if output.outputs else None,
                finish_reason=(
                    output.outputs[0].finish_reason if output.outputs else None
                ),
            )
            for output in raw_outputs
        ]

    def _load_engine(self) -> LLM:
        if self._llm is not None:
            return self._llm
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from vllm import LLM

        logger.info("Loading vLLM model: %s", self.model_path.name)
        self._llm = LLM(
            model=str(self.model_path),
            gpu_memory_utilization=0.7,
            language_model_only=True,
            max_model_len=MODEL_CONTEXT_LENGTH,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            structured_outputs_config={
                "backend": "xgrammar",
                "disable_any_whitespace": True,
            },
        )
        return self._llm

    def _release_engine(self) -> None:
        if self._llm is None:
            return
        self._llm = None
        _ = gc.collect()
        import torch

        torch.cuda.empty_cache()
        logger.info("vLLM model unloaded, GPU memory released.")

    @staticmethod
    def _parse_output(
        request: TranslationRequest,
        output: InferenceResult,
    ) -> TranslationResult:
        if output.text is None:
            raise TranslationOutputError("Translation request returned no output")
        if output.finish_reason != "stop":
            core_range = f"{request.core_ids[0]}-{request.core_ids[-1]}"
            window_range = f"{request.window_ids[0]}-{request.window_ids[-1]}"
            raise TranslationOutputError(
                "Translation request for core cues "
                + f"{core_range} (window {window_range}) stopped with "
                + f"{output.finish_reason!r} after {len(output.text)} characters"
            )
        return parse_translation_output(output.text.strip(), request)

    @staticmethod
    def _reassemble_subtitles(
        subtitle_count: int,
        results: Sequence[TranslationResult],
    ) -> list[str]:
        translations = [""] * subtitle_count
        for result in results:
            stop_index = result.start_index + len(result.texts)
            translations[result.start_index : stop_index] = result.texts
        if any(not text for text in translations):
            raise TranslationOutputError(
                "Translation results do not cover every subtitle"
            )
        return translations
