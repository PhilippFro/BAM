"""PyTorch inference: one forward pass, then skeleton -> CPDAG -> DAG."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_decoding import decode_predictions
from joint_model_torch import JointBAMTorch


def load_model(model_dir, device="cpu"):
    """Rebuild a model from JSON configuration and a tensor-only state dict."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("convention") != "source_target":
        raise ValueError("Expected source_target model convention")
    model = JointBAMTorch(**config["model"])
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model, observations, target="both", threshold=0.5, beam_width=32,
            standardize=True):
    """Return decoded graphs and raw scores for one [samples,variables] matrix.

    The model is temporarily put in evaluation mode, then restored to its
    previous mode. DAG output is a mechanism-dependent point estimate and a
    consistent extension of the decoded CPDAG, not an identifiability claim.
    """
    if isinstance(observations, torch.Tensor):
        x = observations.detach().cpu().numpy()
    else:
        x = np.asarray(observations)
    x = np.array(x, dtype=np.float32, copy=True)
    if x.ndim != 2 or min(x.shape) < 1 or x.shape[0] < 2:
        raise ValueError("observations must have shape [samples>=2, variables>=1]")
    if not np.isfinite(x).all():
        raise ValueError("observations must be finite")
    if standardize:
        x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    parameter = next(model.parameters())
    tensor = torch.from_numpy(x).to(device=parameter.device, dtype=parameter.dtype)
    previous_mode = model.training
    try:
        model.eval()
        with torch.inference_mode():
            outputs = model(tensor.unsqueeze(0))
        scores = {name: value[0].detach().cpu().numpy() for name, value in outputs.items()}
    finally:
        model.train(previous_mode)
    return decode_predictions(scores, target, threshold, beam_width), scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help=".npy matrix or .npz with key X")
    parser.add_argument("--output", type=Path, required=True, help="New .npz file")
    parser.add_argument("--target", choices=("skeleton", "cpdag", "dag", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("--output must be a new .npz file")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            observations = loaded["X"]
    else:
        observations = loaded
    model = load_model(args.model_dir, args.device)
    graphs, scores = predict(model, observations, args.target, args.threshold,
                             args.beam_width, not args.no_standardize)
    arrays = {key: value for key, value in graphs.items() if key != "metadata"}
    arrays.update({"score_" + key: value for key, value in scores.items()})
    arrays["metadata"] = np.asarray(json.dumps(graphs["metadata"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps(graphs["metadata"], indent=2))
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
