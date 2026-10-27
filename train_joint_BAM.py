"""Train one BAM network for CPDAG inference and optional DAG refinement."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from joint_data_BAM import sample_batch
from joint_model_BAM import JointBAM
from joint_training_BAM import make_train_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cpdag", "dag", "both"), default="both")
    parser.add_argument("--mechanisms", nargs="+", default=["linear_gaussian", "nonlinear_anm", "linear_nongaussian"],
                        choices=("linear_gaussian", "nonlinear_anm", "linear_nongaussian", "chebyshev"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--variables", type=int, default=16)
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--data-layers", type=int, default=2)
    parser.add_argument("--cov-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/joint_bam"))
    args = parser.parse_args(argv)
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.samples, args.variables) < 1:
        parser.error("epochs, steps, batch size, samples, and variables must be positive")
    if args.samples < 2 or args.variables < 2:
        parser.error("At least two samples and two variables are required")
    if args.task != "cpdag" and not set(args.mechanisms) & {"nonlinear_anm", "linear_nongaussian"}:
        parser.error("DAG training needs an explicitly supported identifiable mechanism; use --task cpdag otherwise")
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model_config = dict(n_channels=args.channels, hidden_channels=args.hidden_channels,
                        data_layers=args.data_layers, cov_layers=args.cov_layers,
                        n_heads=args.heads, gate_init=args.gate_init,
                        include_dag_heads=args.task != "cpdag")
    model = JointBAM(**model_config)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1.0)
    loss_options = dict(direction_weight=0.0 if args.task == "cpdag" else 1.0,
                        order_weight=0.0 if args.task == "cpdag" else 0.1)
    step = make_train_step(model, optimizer, **loss_options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "config.json").exists():
        parser.error("Output directory already contains a model; choose a new --output-dir")
    config = {"model": model_config, "training": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
              "convention": "source_target", "loss_options": loss_options}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        supervised = 0
        for _ in range(args.steps_per_epoch):
            x, targets = sample_batch(args.batch_size, args.samples, args.variables,
                                      mechanism=args.mechanisms,
                                      expected_degree=args.expected_degree, rng=rng)
            values = step(x, targets)
            epoch_losses.append({name: float(value.numpy()) for name, value in values.items()})
            supervised += int(np.sum(targets["dag_mask"]))
        report = {name: float(np.mean([values[name] for values in epoch_losses]))
                  for name in epoch_losses[0]}
        report.update(epoch=epoch + 1, dag_supervised_datasets=supervised)
        history.append(report)
        print(json.dumps(report), flush=True)
        model.save_weights(str(args.output_dir / "model.weights.h5"))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
