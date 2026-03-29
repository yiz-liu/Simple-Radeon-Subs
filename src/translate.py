import argparse
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pysrt
import requests
from tqdm import tqdm

from src.config import (
    GEMINI_API_KEY,
    GEMINI_API_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL_ID,
    TRANSLATION_PROVIDER,
    VLLM_MODEL_PATH,
)
from src.logger import logger


class BaseTranslator(ABC):
    """Abstract base class for subtitle translators."""

    @abstractmethod
    def translate_chunk(
        self, chunk_id: int, texts: List[str], target_lang: str
    ) -> Dict:
        """Translates a single chunk of subtitles."""
        pass

    def load_subtitles(self, path: Path) -> pysrt.SubRipFile:
        """Loads subtitles from a file."""
        logger.info("Loading %s...", path.name)
        try:
            return pysrt.open(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load SRT: {e}") from e

    def prepare_chunks(self, subs: pysrt.SubRipFile, batch_size: int) -> List[Dict]:
        """Splits subtitles into batches for translation."""
        total_subs = len(subs)
        chunks = []
        for i in range(0, total_subs, batch_size):
            chunk_texts = [
                sub.text.replace("\n", " ") for sub in subs[i : i + batch_size]
            ]
            chunks.append({"id": i, "texts": chunk_texts})
        return chunks

    def execute_concurrent_translation(
        self, chunks: List[Dict], target_lang: str, workers: int
    ) -> Dict[int, List[str]]:
        """Executes translation tasks in parallel and returns a map of results."""
        translated_map = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self.translate_chunk, chunk["id"], chunk["texts"], target_lang
                ): chunk
                for chunk in chunks
            }

            with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:
                for future in as_completed(future_to_chunk):
                    result = future.result()
                    translated_map[result["id"]] = result["lines"]
                    pbar.update(1)

        return translated_map

    def reassemble_subtitles(
        self, subs: pysrt.SubRipFile, translated_map: Dict[int, List[str]]
    ) -> List[str]:
        """Returns translated text aligned to the cleaned subtitle order."""
        logger.info("Reassembling subtitle file...")
        final_translations = [""] * len(subs)
        for start_idx in sorted(translated_map.keys()):
            for offset, translated_text in enumerate(translated_map[start_idx]):
                target_idx = start_idx + offset
                if target_idx < len(final_translations):
                    final_translations[target_idx] = translated_text

        return final_translations

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        target_lang: str = "Chinese",
        batch_size: int = 128,
        workers: int = 16,
        translated_only: bool = False,
    ):
        """Orchestrates the entire translation process."""
        subs = self.load_subtitles(input_path)

        if len(subs) == 0:
            logger.warning("No subtitles found to translate.")
            subs.save(str(output_path), encoding="utf-8")
            return

        chunks = self.prepare_chunks(subs, batch_size)

        logger.info(
            "Translating %d segments (%d chunks, %d threads).",
            len(subs),
            len(chunks),
            workers,
        )

        translated_map = self.execute_concurrent_translation(
            chunks, target_lang, workers
        )

        translated_lines = self.reassemble_subtitles(subs, translated_map)

        self._save_filtered(subs, translated_lines, output_path, translated_only)

    def _build_rules(self, count: int) -> str:
        """Builds the shared translation rules prompt fragment."""
        return (
            f"**Rules:**\n"
            f"1. **Alignment**: Output exactly {count} lines, one per line, no numbering. "
            f"Line N maps to Input N. Use `[SKIP]` as a placeholder for any skipped line.\n"
            f"2. **Skip Fillers**: Output `[SKIP]` for lines that are pure non-linguistic "
            f'vocalizations or interjections with no translatable meaning (e.g. "hmm", "ugh", "うー").\n'
            f"3. **Skip Garbage**: Output `[SKIP]` for lines that are garbled or hallucinated "
            f"(incoherent mix of scripts/languages, random characters with no linguistic meaning).\n"
            f"4. **No Extras**: Output only the translated text. No explanations, notes, or original text."
        )

    def _parse_translation_output(
        self, raw_text: str, expected_count: int
    ) -> List[str]:
        """Parses raw LLM translation response into a count-aligned list of strings."""
        lines = raw_text.splitlines()
        translated_lines: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            clean_line = re.sub(r"^\[?\d+\]?\s*:?\s*", "", line)
            if clean_line.strip().upper() in ("[SKIP]", "SKIP"):
                translated_lines.append("")
            else:
                translated_lines.append(clean_line)
        if len(translated_lines) < expected_count:
            translated_lines += [""] * (expected_count - len(translated_lines))
        elif len(translated_lines) > expected_count:
            translated_lines = translated_lines[:expected_count]
        return translated_lines

    def _build_chat_messages(
        self, texts: List[str], target_lang: str
    ) -> List[Dict[str, str]]:
        """Builds standard system+user messages for chat-format APIs."""
        count = len(texts)
        system_prompt = (
            f"You are a professional movie subtitle translator. "
            f"Translate the following {count} subtitle segment(s) into {target_lang}.\n\n"
            + self._build_rules(count)
        )
        user_prompt = "\n".join([f"[{idx + 1}] {t}" for idx, t in enumerate(texts)])
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _save_filtered(
        self,
        subs: pysrt.SubRipFile,
        translated_lines: List[str],
        output_path: Path,
        translated_only: bool,
    ) -> None:
        """Filters empty translated lines, re-indexes, and saves the final subtitle file."""
        final_items = []

        for sub, translated_text in zip(subs, translated_lines):
            if not translated_text.strip():
                continue

            final_text = translated_text
            if not translated_only:
                source_text = sub.text.strip()
                final_text = (
                    f"{source_text}\n{translated_text}" if source_text else translated_text
                )

            final_items.append(
                pysrt.SubRipItem(
                    index=len(final_items) + 1,
                    start=sub.start,
                    end=sub.end,
                    text=final_text,
                    position=sub.position,
                )
            )

        clean_subs = pysrt.SubRipFile(items=final_items)
        logger.info("Saving to %s...", output_path.name)
        clean_subs.save(str(output_path), encoding="utf-8")


class GeminiTranslator(BaseTranslator):
    """Uses Google Gemini API to translate subtitle segments concurrently."""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url

    def translate_chunk(
        self, chunk_id: int, texts: List[str], target_lang: str = "Chinese"
    ) -> Dict:
        if not texts:
            return {"id": chunk_id, "lines": []}

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")

        count = len(texts)
        prompt = (
            f"You are a professional movie subtitle translator. "
            f"Translate the following {count} subtitle segments into {target_lang}.\n\n"
            + self._build_rules(count)
            + "\n\n"
            + "\n".join([f"[{idx + 1}] {t}" for idx, t in enumerate(texts)])
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        url = f"{self.api_url}?key={self.api_key}"

        max_retries = 5
        base_delay = 2
        response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code in [429, 500, 502, 503, 504]:
                    response.raise_for_status()
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error("Chunk %d failed: %s", chunk_id, e)
                    return {"id": chunk_id, "lines": [f"[ERROR] {t}" for t in texts]}
                time.sleep(base_delay * (2**attempt))

        if response is None:
            return {"id": chunk_id, "lines": [f"[ERROR] {t}" for t in texts]}

        try:
            data = response.json()
            if "candidates" not in data or not data["candidates"]:
                return {"id": chunk_id, "lines": [f"[API ERROR] {t}" for t in texts]}

            part = data["candidates"][0]["content"]["parts"][0]
            translated_text = part.get("text", "").strip()

            return {
                "id": chunk_id,
                "lines": self._parse_translation_output(translated_text, count),
            }

        except Exception as e:
            logger.error("Chunk %d parsing error: %s", chunk_id, e)
            return {"id": chunk_id, "lines": [f"[PARSE ERROR] {t}" for t in texts]}


class OpenAITranslator(BaseTranslator):
    """Uses OpenAI-compatible API to translate subtitle segments concurrently."""

    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id

    def translate_chunk(
        self, chunk_id: int, texts: List[str], target_lang: str = "Chinese"
    ) -> Dict:
        if not texts:
            return {"id": chunk_id, "lines": []}

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")

        count = len(texts)
        messages = self._build_chat_messages(texts, target_lang)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.2,
        }

        max_retries = 5
        base_delay = 2
        response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=60
                )
                if response.status_code in [429, 500, 502, 503, 504]:
                    response.raise_for_status()
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error("Chunk %d failed: %s", chunk_id, e)
                    return {"id": chunk_id, "lines": [f"[ERROR] {t}" for t in texts]}
                time.sleep(base_delay * (2**attempt))

        if response is None:
            return {"id": chunk_id, "lines": [f"[ERROR] {t}" for t in texts]}

        try:
            data = response.json()
            if "choices" not in data or not data["choices"]:
                return {"id": chunk_id, "lines": [f"[API ERROR] {t}" for t in texts]}

            translated_text = data["choices"][0]["message"]["content"].strip()

            return {
                "id": chunk_id,
                "lines": self._parse_translation_output(translated_text, count),
            }

        except Exception as e:
            logger.error("Chunk %d parsing error: %s", chunk_id, e)
            return {"id": chunk_id, "lines": [f"[PARSE ERROR] {t}" for t in texts]}


class VLLMTranslator(BaseTranslator):
    """Uses vLLM offline batch inference to translate subtitle segments locally.

    Loads the model lazily inside translate_srt() to allow GPU memory to be
    freed between Whisper transcription and translation steps.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"vLLM model not found: {model_path}")
        self._llm = None
        self._sampling_params = None

    def _ensure_llm_loaded(self) -> None:
        """Lazily initializes the vLLM engine. Idempotent."""
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        logger.info("Loading vLLM model: %s", self.model_path.name)
        self._llm = LLM(
            model=str(self.model_path),
            dtype="float16",
            gpu_memory_utilization=0.75,
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

    def _release_llm(self) -> None:
        """Frees GPU memory. Safe to call even if never loaded."""
        if self._llm is None:
            return
        del self._llm
        self._llm = None
        self._sampling_params = None
        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("vLLM model unloaded, GPU memory released.")

    def translate_chunk(
        self, chunk_id: int, texts: List[str], target_lang: str
    ) -> Dict:
        """Translates one chunk via vLLM. Requires _ensure_llm_loaded() to have been called."""
        if not texts:
            return {"id": chunk_id, "lines": []}
        if self._llm is None:
            raise RuntimeError(
                "VLLMTranslator._llm is not loaded. "
                "Call translate_srt() which manages the lifecycle automatically."
            )
        conversation = self._build_chat_messages(texts, target_lang)
        outputs = self._llm.chat(
            [conversation],  # type: ignore[arg-type]  # vLLM accepts List[List[Dict]]
            sampling_params=self._sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        raw_text = outputs[0].outputs[0].text.strip()
        if outputs[0].outputs[0].finish_reason == "length":
            logger.warning("Chunk %d truncated (hit max_tokens limit)", chunk_id)
        # Defensive: strip any thinking tokens that leaked (Qwen3 may emit them)
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # Reuse BaseTranslator._parse_translation_output
        return {
            "id": chunk_id,
            "lines": self._parse_translation_output(raw_text, len(texts)),
        }

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        target_lang: str = "Chinese",
        # Default 16: constrained by personal device VRAM; avoids exceeding context window
        batch_size: int = 16,
        workers: int = 16,
        translated_only: bool = False,
    ):
        """Translates subtitles using vLLM offline batch inference.

        Sends batch_size subtitles per LLM conversation. vLLM's internal scheduler
        (max_num_seqs) handles parallelism; no ThreadPoolExecutor needed.
        """
        subs = self.load_subtitles(input_path)

        if len(subs) == 0:
            logger.warning("No subtitles found to translate.")
            subs.save(str(output_path), encoding="utf-8")
            return

        # Reuse base class prepare_chunks
        chunks = self.prepare_chunks(subs, batch_size)

        logger.info(
            "Translating %d segments (%d chunks of %d) via vLLM (model: %s).",
            len(subs),
            len(chunks),
            batch_size,
            self.model_path.name,
        )

        # Build all conversations (one per chunk)
        conversations = [
            self._build_chat_messages(chunk["texts"], target_lang) for chunk in chunks
        ]

        self._ensure_llm_loaded()
        assert self._llm is not None  # guaranteed by _ensure_llm_loaded
        try:
            outputs = self._llm.chat(
                conversations,  # type: ignore[arg-type]  # vLLM accepts List[List[Dict]]
                sampling_params=self._sampling_params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=True,
            )
        finally:
            self._release_llm()

        # Parse results into translated_map (matches BaseTranslator.reassemble_subtitles format)
        translated_map: Dict[int, List[str]] = {}
        for chunk, output in zip(chunks, outputs):
            raw_text = output.outputs[0].text.strip()
            if output.outputs[0].finish_reason == "length":
                logger.warning(
                    "Chunk starting at subtitle %d truncated (hit max_tokens limit)",
                    chunk["id"] + 1,
                )
            # Defensive: strip any thinking tokens
            raw_text = re.sub(
                r"<think>.*?</think>", "", raw_text, flags=re.DOTALL
            ).strip()
            # Reuse BaseTranslator._parse_translation_output
            translated_map[chunk["id"]] = self._parse_translation_output(
                raw_text, len(chunk["texts"])
            )

        # Reuse base class methods for reassembly and saving
        translated_lines = self.reassemble_subtitles(subs, translated_map)
        self._save_filtered(subs, translated_lines, output_path, translated_only)


def create_translator(provider: str | None = None) -> BaseTranslator:
    """Factory function to create translator based on provider."""
    provider = provider or TRANSLATION_PROVIDER

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        logger.info("Using OpenAI-compatible translator (model: %s)", OPENAI_MODEL_ID)
        return OpenAITranslator(OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_ID)
    elif provider == "vllm":
        if not VLLM_MODEL_PATH:
            raise ValueError("VLLM_MODEL_PATH is not set in .env")
        logger.info("Using vLLM offline translator (model: %s)", VLLM_MODEL_PATH)
        return VLLMTranslator(VLLM_MODEL_PATH)
    else:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        logger.info("Using Gemini translator")
        return GeminiTranslator(GEMINI_API_KEY, GEMINI_API_URL)


def main():
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using Gemini or OpenAI-compatible API (Concurrent)."
    )
    parser.add_argument("input", help="Path to the input SRT file.")
    parser.add_argument("-o", "--output", help="Path to the output SRT file.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Lines per HTTP Request (Context window size). Default: 128.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel requests. Default: 16.",
    )
    parser.add_argument("--lang", default="Chinese", help="Target language.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "vllm"],
        default=None,
        help="Translation provider (overrides TRANSLATION_PROVIDER env var).",
    )
    parser.add_argument(
        "--translated-only",
        action="store_true",
        help="Generate translated subtitles only instead of bilingual subtitles.",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()

    suffix = f".{args.lang.lower().replace(' ', '_')}.srt"
    output_path = Path(
        args.output or input_path.parent / (input_path.stem + suffix)
    ).resolve()

    if not input_path.exists():
        logger.error("Error: %s not found.", input_path)
        exit(1)

    translator = create_translator(args.provider)
    translator.translate_srt(
        input_path,
        output_path,
        target_lang=args.lang,
        batch_size=args.batch_size,
        workers=args.workers,
        translated_only=args.translated_only,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
