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
    ):
        """Updates the subtitle objects with translated text in-place."""
        logger.info("Reassembling subtitle file...")
        final_translations = []
        for start_idx in sorted(translated_map.keys()):
            final_translations.extend(translated_map[start_idx])

        for i, sub in enumerate(subs):
            if i < len(final_translations):
                sub.text = final_translations[i]

    def translate_srt(
        self,
        input_path: Path,
        output_path: Path,
        target_lang: str = "Chinese",
        batch_size: int = 128,
        workers: int = 16,
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

        self.reassemble_subtitles(subs, translated_map)

        non_empty_items = [sub for sub in subs if sub.text.strip()]
        for i, sub in enumerate(non_empty_items, 1):
            sub.index = i
        clean_subs = pysrt.SubRipFile(items=non_empty_items)

        logger.info("Saving to %s...", output_path.name)
        clean_subs.save(str(output_path), encoding="utf-8")

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

            lines = translated_text.splitlines()
            translated_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                clean_line = re.sub(r"^\[?\d+\]?\s*:?\s*", "", line)
                if clean_line.strip().upper() in ("[SKIP]", "SKIP"):
                    translated_lines.append("")
                else:
                    translated_lines.append(clean_line)

            if len(translated_lines) < count:
                translated_lines += [""] * (count - len(translated_lines))
            elif len(translated_lines) > count:
                translated_lines = translated_lines[:count]

            return {"id": chunk_id, "lines": translated_lines}

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
        system_prompt = (
            f"You are a professional movie subtitle translator. "
            f"Translate subtitle segments into {target_lang}.\n\n"
            + self._build_rules(count)
        )

        user_prompt = "\n".join([f"[{idx + 1}] {t}" for idx, t in enumerate(texts)])

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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

            lines = translated_text.splitlines()
            translated_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                clean_line = re.sub(r"^\[?\d+\]?\s*:?\s*", "", line)
                if clean_line.strip().upper() in ("[SKIP]", "SKIP"):
                    translated_lines.append("")
                else:
                    translated_lines.append(clean_line)

            if len(translated_lines) < count:
                translated_lines += [""] * (count - len(translated_lines))
            elif len(translated_lines) > count:
                translated_lines = translated_lines[:count]

            return {"id": chunk_id, "lines": translated_lines}

        except Exception as e:
            logger.error("Chunk %d parsing error: %s", chunk_id, e)
            return {"id": chunk_id, "lines": [f"[PARSE ERROR] {t}" for t in texts]}


def create_translator(provider: str | None = None) -> BaseTranslator:
    """Factory function to create translator based on provider."""
    provider = provider or TRANSLATION_PROVIDER

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        logger.info("Using OpenAI-compatible translator (model: %s)", OPENAI_MODEL_ID)
        return OpenAITranslator(OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_ID)
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
        choices=["gemini", "openai"],
        default=None,
        help="Translation provider (overrides TRANSLATION_PROVIDER env var).",
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
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
