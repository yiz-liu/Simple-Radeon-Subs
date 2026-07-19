from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

import pysrt

from src.config import TRANSLATION_MODEL_PATH
from src.logger import logger

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
    from vllm.outputs import RequestOutput

DEFAULT_BATCH_SIZE: Final = 16


@dataclass(frozen=True, slots=True)
class TranslationOptions:
    """Options shared by full-pipeline and standalone translation."""

    target_lang: str = "Chinese"
    batch_size: int = DEFAULT_BATCH_SIZE
    translated_only: bool = False


DEFAULT_TRANSLATION_OPTIONS: Final = TranslationOptions()


@dataclass(frozen=True, slots=True)
class TranslationChunk:
    start_index: int
    texts: tuple[str, ...]


@final
class VLLMTranslator:
    """Translate subtitle batches with the managed local vLLM model."""

    def __init__(self, model_path: Path = TRANSLATION_MODEL_PATH) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"vLLM model not found: {self.model_path}")
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        options: TranslationOptions = DEFAULT_TRANSLATION_OPTIONS,
    ) -> None:
        """Translate an SRT file and preserve the source timing information."""
        subs = self._load_subtitles(input_path)
        if not subs:
            logger.warning("No subtitles found to translate.")
            subs.save(str(output_path), encoding="utf-8")
            return

        chunks = self._prepare_chunks(subs, options.batch_size)
        logger.info(
            "Translating %d segments (%d chunks of %d) via vLLM (model: %s).",
            len(subs),
            len(chunks),
            options.batch_size,
            self.model_path.name,
        )
        conversations = [
            self._build_chat_messages(chunk.texts, options.target_lang)
            for chunk in chunks
        ]
        try:
            outputs = self._run_inference(conversations)
        finally:
            self._release_engine()

        translated_map: dict[int, list[str]] = {}
        for chunk, output in zip(chunks, outputs, strict=True):
            generated = output.outputs[0]
            if generated.finish_reason == "length":
                logger.warning(
                    "Translation chunk truncated at subtitle %d.",
                    chunk.start_index + 1,
                )
            raw_text = re.sub(
                r"<think>.*?</think>",
                "",
                generated.text.strip(),
                flags=re.DOTALL,
            ).strip()
            translated_map[chunk.start_index] = self._parse_translation_output(
                raw_text,
                len(chunk.texts),
            )

        translated_lines = self._reassemble_subtitles(subs, translated_map)
        self._save_filtered(
            subs,
            translated_lines,
            output_path,
            options.translated_only,
        )

    def _run_inference(
        self,
        conversations: list[list[ChatCompletionMessageParam]],
    ) -> list[RequestOutput]:
        llm, sampling_params = self._load_engine()
        return llm.chat(
            conversations,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )

    def _load_engine(self) -> tuple[LLM, SamplingParams]:
        if self._llm is not None and self._sampling_params is not None:
            return self._llm, self._sampling_params

        from vllm import LLM, SamplingParams

        logger.info("Loading vLLM model: %s", self.model_path.name)
        self._llm = LLM(
            model=str(self.model_path),
            dtype="float16",
            gpu_memory_utilization=0.6,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            enable_prefix_caching=True,
        )
        self._sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            max_tokens=1024,
            skip_special_tokens=True,
        )
        return self._llm, self._sampling_params

    def _release_engine(self) -> None:
        if self._llm is None:
            return
        self._llm = None
        self._sampling_params = None
        _ = gc.collect()

        import torch

        torch.cuda.empty_cache()
        logger.info("vLLM model unloaded, GPU memory released.")

    @staticmethod
    def _load_subtitles(path: Path) -> pysrt.SubRipFile:
        logger.info("Loading %s...", path.name)
        return pysrt.open(str(path.resolve()))

    @staticmethod
    def _prepare_chunks(
        subs: pysrt.SubRipFile,
        batch_size: int,
    ) -> list[TranslationChunk]:
        return [
            TranslationChunk(
                start_index=start_index,
                texts=tuple(
                    sub.text.replace("\n", " ")
                    for sub in subs[start_index : start_index + batch_size]
                ),
            )
            for start_index in range(0, len(subs), batch_size)
        ]

    @staticmethod
    def _reassemble_subtitles(
        subs: pysrt.SubRipFile,
        translated_map: dict[int, list[str]],
    ) -> list[str]:
        final_translations = [""] * len(subs)
        for start_index in sorted(translated_map):
            for offset, translated_text in enumerate(translated_map[start_index]):
                target_index = start_index + offset
                if target_index < len(final_translations):
                    final_translations[target_index] = translated_text
        return final_translations

    @staticmethod
    def _build_rules(count: int) -> str:
        return (
            "**Rules:**\n"
            f"1. **Alignment**: Output exactly {count} lines, one per line, no numbering. "
            "Line N maps to Input N. Use `[SKIP]` as a placeholder for any skipped line.\n"
            "2. **Skip Fillers**: Output `[SKIP]` for lines that are pure "
            "non-linguistic vocalizations or interjections with no translatable meaning.\n"
            "3. **Skip Garbage**: Output `[SKIP]` for lines that are garbled, "
            "hallucinated, or have no linguistic meaning.\n"
            "4. **No Extras**: Output only translated text without explanations, "
            "notes, or original text."
        )

    @staticmethod
    def _parse_translation_output(raw_text: str, expected_count: int) -> list[str]:
        translated_lines: list[str] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            clean_line = re.sub(r"^\[?\d+\]?\s*:?\s*", "", line)
            translated_lines.append(
                "" if clean_line.strip().upper() in ("[SKIP]", "SKIP") else clean_line
            )
        if len(translated_lines) < expected_count:
            translated_lines.extend([""] * (expected_count - len(translated_lines)))
        return translated_lines[:expected_count]

    def _build_chat_messages(
        self,
        texts: tuple[str, ...],
        target_lang: str,
    ) -> list[ChatCompletionMessageParam]:
        count = len(texts)
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional movie subtitle translator. "
                    f"Translate the following {count} subtitle segment(s) into "
                    f"{target_lang}.\n\n{self._build_rules(count)}"
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    f"[{index}] {text}" for index, text in enumerate(texts, start=1)
                ),
            },
        ]

    @staticmethod
    def _save_filtered(
        subs: pysrt.SubRipFile,
        translated_lines: list[str],
        output_path: Path,
        translated_only: bool,
    ) -> None:
        final_items: list[pysrt.SubRipItem] = []
        for sub, translated_text in zip(subs, translated_lines, strict=True):
            if not translated_text.strip():
                continue
            source_text = sub.text.strip()
            final_text = translated_text
            if not translated_only and source_text:
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
