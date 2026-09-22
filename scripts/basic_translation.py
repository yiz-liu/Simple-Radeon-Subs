#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATION_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-9B-AWQ-4bit"
MODEL_CONTEXT_LENGTH = 4096
MAX_TRANSLATION_TOKENS = 2048

BATCH = (
    (
        "Les Misérables",
        (
            "Il fait nuit, et pourtant la ville ne dort pas.",
            "L'homme qui marche porte toujours une espérance.",
            "Aimer, c'est agir.",
        ),
    ),
    (
        "Notre-Dame de Paris",
        (
            "Ceci tuera cela.",
            "Il y a des âmes qui ont besoin de regarder le ciel.",
            "Le temps est un grand sculpteur.",
        ),
    ),
    (
        "Le Comte de Monte-Cristo",
        (
            "Attendre et espérer.",
            "Toute sagesse humaine est résumée dans ces deux mots.",
            "La haine est aveugle, la colère est sourde.",
        ),
    ),
    (
        "Candide",
        (
            "Il faut cultiver notre jardin.",
            "Le travail éloigne de nous trois grands maux.",
            "Tout est au mieux dans le meilleur des mondes possibles.",
        ),
    ),
)


def main() -> None:
    """Run four batched translation requests with three French subtitles each."""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    model_path = TRANSLATION_MODEL_PATH.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Translation model not found: {model_path}")

    request_schema = {
        "translations": [
            {"index": index, "text": "string"} for index in range(3)
        ],
    }
    conversations = [
        [
            {
                "role": "user",
                "content": (
                    "Translate these French subtitles to Chinese. "
                    "Return only JSON matching the requested schema.\n"
                    f"Source work: {work}\n"
                    "Source language: French\n"
                    "Target language: Chinese\n"
                    + "\n".join(
                        f"Subtitle {index}: {subtitle}"
                        for index, subtitle in enumerate(subtitles)
                    )
                ),
            }
        ]
        for work, subtitles in BATCH
    ]

    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=0.7,
        language_model_only=True,
        max_model_len=MODEL_CONTEXT_LENGTH,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        structured_outputs_config={
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        },
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
        max_tokens=MAX_TRANSLATION_TOKENS,
        skip_special_tokens=True,
        structured_outputs=StructuredOutputsParams(
            json=json.dumps(request_schema, separators=(",", ":"))
        ),
    )
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    if len(outputs) != len(BATCH) or any(not output.outputs for output in outputs):
        raise RuntimeError("vLLM returned an incomplete batch")
    for request_index, output in enumerate(outputs, start=1):
        print(f"Request {request_index}: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
