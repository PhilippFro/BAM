"""One neural pass, then skeleton -> CPDAG -> mechanism-dependent DAG."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from joint_decoding import decode_predictions




def load_model(model_dir):
    """Load the explicit architecture config and weights (no arbitrary objects)."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from joint_model_BAM import JointBAM
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("convention") != "source_target":
        raise ValueError("Expected source_target model convention")
    model = JointBAM(**config["model"])
    model(np.zeros((1, 4, 3), dtype=np.float32))
    model.load_weights(str(model_dir / "model.weights.h5"))
    return model


def predict(model, observations, target="both", threshold=0.5, beam_width=32,
            standardize=True):
    """Infer one observed dataset; returned scores are conditional predictions."""
    x = np.asarray(observations, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or x.shape[0] < 2:
        raise ValueError("observations must have shape [samples>=2, variables>=1]")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite")
    if standardize:
        x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    outputs = model(x[None], training=False)
    scores = {name: np.asarray(value)[0] for name, value in outputs.items()}
    return decode_predictions(scores, target, threshold, beam_width), scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help=".npy [samples,variables] or .npz with key X")
    parser.add_argument("--output", type=Path, required=True, help="New .npz file")
    parser.add_argument("--target", choices=("skeleton", "cpdag", "dag", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error("Output file already exists; choose a new path")
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            x = loaded["X"]
    else:
        x = loaded
    model = load_model(args.model_dir)
    result, scores = predict(model, x, args.target, args.threshold, args.beam_width,
                             standardize=not args.no_standardize)
    arrays = {key: value for key, value in result.items() if key != "metadata"}
    arrays.update({"score_" + key: value for key, value in scores.items()})
    arrays["metadata"] = np.asarray(json.dumps(result["metadata"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(result["metadata"], indent=2))
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
