"""Command-line interface for prediction, training, calibration, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import StateCache
from .calibration import TemperatureCalibrator
from .encoders import ClapAudioEncoder, MediaEncoder, SiglipVisionEncoder
from .engine import DecisionEngine
from .head import DynamicDecisionHead
from .schemas import Question
from .training import DecisionHeadTrainer, load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-mm", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="answer decision cards about one media file")
    _add_encoder_arguments(predict)
    predict.add_argument("media")
    predict.add_argument("--questions", required=True, help="JSON file containing questions")
    predict.add_argument("--checkpoint")
    predict.add_argument("--calibration")
    predict.add_argument("--cache-dir")
    predict.add_argument("--abstain-below", type=float)

    encode = subparsers.add_parser("encode", help="save a reusable encoded state")
    _add_encoder_arguments(encode)
    encode.add_argument("media")
    encode.add_argument("--output", required=True)

    train = subparsers.add_parser("train", help="train the dynamic decision head")
    _add_encoder_arguments(train)
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--cache-dir", default=".cache/laya-multimodal")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--loss", choices=["log", "brier", "spherical", "rps"], default="log")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-shuffle-options", action="store_true")

    calibrate = subparsers.add_parser("calibrate", help="fit temperature on held-out data")
    _add_encoder_arguments(calibrate)
    calibrate.add_argument("--dataset", required=True)
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--cache-dir", default=".cache/laya-multimodal")

    evaluate = subparsers.add_parser("evaluate", help="report accuracy and calibration metrics")
    _add_encoder_arguments(evaluate)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--calibration")
    evaluate.add_argument("--cache-dir", default=".cache/laya-multimodal")
    return parser


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modality", choices=["image", "audio"], required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")


def create_encoder(args: argparse.Namespace) -> MediaEncoder:
    if args.modality == "image":
        return SiglipVisionEncoder(
            args.model_id or "google/siglip2-base-patch16-224", device=args.device
        )
    return ClapAudioEncoder(args.model_id or "laion/clap-htsat-fused", device=args.device)


def validate_checkpoint(checkpoint: str | Path, encoder: MediaEncoder) -> None:
    manifest_path = Path(checkpoint) / "encoder.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (manifest.get("modality"), manifest.get("model_id"))
    actual = (encoder.modality, encoder.model_id)
    if expected != actual:
        raise ValueError(
            "checkpoint encoder mismatch: "
            f"checkpoint expects {expected[0]} model {expected[1]!r}, "
            f"but the CLI loaded {actual[0]} model {actual[1]!r}"
        )


def _load_questions(path: str | Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if isinstance(data, dict):
        return [Question.from_dict({"id": key, **value}) for key, value in data.items()]
    return [Question.from_dict(value) for value in data]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = create_encoder(args)
    if args.command == "encode":
        encoder.encode(args.media).detach(cpu=True).save(args.output)
        print(args.output)
        return 0

    head = None
    if getattr(args, "checkpoint", None):
        validate_checkpoint(args.checkpoint, encoder)
        head = DynamicDecisionHead.from_pretrained(args.checkpoint, device=encoder.device)

    if args.command == "predict":
        calibrator = (
            TemperatureCalibrator.load(args.calibration)
            if args.calibration
            else TemperatureCalibrator()
        )
        engine = DecisionEngine(
            encoder,
            decision_head=head,
            calibrator=calibrator,
            abstain_below=args.abstain_below,
            cache=StateCache(args.cache_dir) if args.cache_dir else None,
        )
        result = engine.predict(args.media, _load_questions(args.questions))
        print(json.dumps({key: value.to_dict() for key, value in result.items()}, indent=2))
        return 0

    examples = load_jsonl(args.dataset)
    trainer = DecisionHeadTrainer(
        encoder,
        head,
        cache_directory=args.cache_dir,
        seed=getattr(args, "seed", 42),
    )
    if args.command == "train":
        trainer.train(
            examples,
            output_directory=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss=args.loss,
            shuffle_options=not args.no_shuffle_options,
        )
        return 0
    if args.command == "calibrate":
        calibrator = trainer.calibrate(examples, output_path=args.output)
        print(
            json.dumps(
                {"default": calibrator.default, "by_cardinality": calibrator.by_cardinality},
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate":
        calibrator = TemperatureCalibrator.load(args.calibration) if args.calibration else None
        print(json.dumps(trainer.evaluate(examples, calibrator=calibrator), indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
