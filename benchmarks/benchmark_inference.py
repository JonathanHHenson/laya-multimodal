"""Measure cold encoding and warm decision latency on the current machine."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from laya_multimodal import DecisionEngine, Question
from laya_multimodal.encoders import ClapAudioEncoder, SiglipVisionEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "audio"], required=True)
    parser.add_argument("--media", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    encoder = (
        SiglipVisionEncoder(device=args.device)
        if args.modality == "image"
        else ClapAudioEncoder(device=args.device)
    )
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [Question.from_dict({"id": key, **value}) for key, value in data.items()]
    engine = DecisionEngine(encoder)

    started = time.perf_counter()
    state = engine.encode(args.media, use_cache=False)
    encode_seconds = time.perf_counter() - started
    engine.ask(state, questions)  # warm up kernels
    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        engine.ask(state, questions)
        timings.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "device": str(encoder.device),
                "model": encoder.model_id,
                "question_count": len(questions),
                "encode_seconds": encode_seconds,
                "ask_mean_seconds": statistics.mean(timings),
                "ask_p50_seconds": statistics.median(timings),
                "ask_min_seconds": min(timings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
