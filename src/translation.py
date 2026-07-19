from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

import pysrt

from src.config import TRANSLATION_MODEL_PATH
from src.logger import logger
from src.translation_requests import (
    TranslationOutputError,
    TranslationRequest,
    TranslationResult,
    build_translation_requests,
    parse_translation_output,
)

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.outputs import RequestOutput

MODEL_CONTEXT_LENGTH: Final = 4096


@dataclass(frozen=True, slots=True)
class TranslationOptions:
    """Options shared by full-pipeline and standalone translation."""

    target_lang: str = "Chinese"
    translated_only: bool = False


DEFAULT_TRANSLATION_OPTIONS: Final = TranslationOptions()


@final
class VLLMTranslator:
    """Translate subtitles with the managed local vLLM model."""

    def __init__(self, model_path: Path = TRANSLATION_MODEL_PATH) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"vLLM model not found: {self.model_path}")
        self._llm: LLM | None = None

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        options: TranslationOptions = DEFAULT_TRANSLATION_OPTIONS,
    ) -> None:
        """Translate an SRT file while preserving every source timestamp."""
        logger.info("Loading %s...", input_path.name)
        subs = pysrt.open(str(input_path.resolve()))
        if not subs:
            logger.warning("No subtitles found to translate.")
            subs.save(str(output_path), encoding="utf-8")
            return

        requests = build_translation_requests(
            tuple(sub.text.replace("\n", " ") for sub in subs)
        )
        logger.info(
            "Translating %d segments in %d overlapping requests via vLLM (model: %s).",
            len(subs),
            len(requests),
            self.model_path.name,
        )
        try:
            outputs = self._run_inference(requests, options.target_lang)
            results = self._parse_outputs(requests, outputs)
        finally:
            self._release_engine()

        translated_lines = self._reassemble_subtitles(len(subs), results)
        final_items: list[pysrt.SubRipItem] = []
        for sub, translated_text in zip(subs, translated_lines, strict=True):
            source_text = sub.text.strip()
            final_text = translated_text
            if not options.translated_only and source_text:
                final_text = f"{source_text}\n{translated_text}"
            final_items.append(
                pysrt.SubRipItem(
                    index=len(final_items) + 1,
                    start=sub.start,
                    end=sub.end,
                    text=final_text,
                    position=sub.position,
                )
            )

        logger.info("Saving to %s...", output_path.name)
        pysrt.SubRipFile(items=final_items).save(
            str(output_path.resolve()),
            encoding="utf-8",
        )

    def _run_inference(
        self,
        requests: Sequence[TranslationRequest],
        target_lang: str,
    ) -> list[RequestOutput]:
        llm = self._load_engine()

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sampling_params = [
            SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                presence_penalty=1.5,
                max_tokens=None,
                skip_special_tokens=True,
                structured_outputs=StructuredOutputsParams(
                    json=json.dumps(request.schema, separators=(",", ":"))
                ),
            )
            for request in requests
        ]
        return llm.chat(
            [request.messages(target_lang) for request in requests],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )

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
    def _parse_outputs(
        requests: Sequence[TranslationRequest],
        outputs: Sequence[RequestOutput],
    ) -> list[TranslationResult]:
        results: list[TranslationResult] = []
        for request, output in zip(requests, outputs, strict=True):
            generated = output.outputs[0]
            if generated.finish_reason != "stop":
                raise TranslationOutputError(
                    f"Translation request stopped with {generated.finish_reason!r}"
                )
            results.append(parse_translation_output(generated.text.strip(), request))
        return results

    @staticmethod
    def _reassemble_subtitles(
        subtitle_count: int,
        results: Sequence[TranslationResult],
    ) -> list[str]:
        translations = [""] * subtitle_count
        for result in results:
            stop_index = result.start_index + len(result.texts)
            translations[result.start_index:stop_index] = result.texts
        if any(not text for text in translations):
            raise TranslationOutputError("Translation results do not cover every subtitle")
        return translations
