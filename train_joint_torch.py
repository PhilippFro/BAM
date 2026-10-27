"""Train PyTorch BAM, or check the entire pipeline with --smoke-test."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from generator_torch import DataGeneratorBAM
from graph_utils_BAM import dag_to_cpdag, validate_dag
from joint_model_torch import JointBAMTorch
from joint_training_torch import joint_losses, train_step
from predict_joint_torch import load_model, predict


def _device_batch(batch, device):
    observations, targets = batch
    return observations.to(device), {name: value.to(device) for name, value in targets.items()}


def _scalars(losses):
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def _validate_checkpoint(model, checkpoint, observations, cpdag_only=False):
    """Load saved state and verify scores plus all requested decoding stages."""
    device = next(model.parameters()).device
    restored = load_model(checkpoint.parent, device=device)
    model.eval()
    restored.eval()
    with torch.no_grad():
        original = model(observations)
        reloaded = restored(observations)
    errors = []
    for name in original:
        torch.testing.assert_close(original[name], reloaded[name], rtol=1e-6, atol=1e-7)
        errors.append(float((original[name] - reloaded[name]).abs().max().cpu()))
    graphs, _ = predict(restored, observations[0], target="cpdag" if cpdag_only else "both",
                         beam_width=16, standardize=False)
    cpdag = graphs["cpdag"]
    report = {"checkpoint_max_error": max(errors), "cpdag_decoded": True}
    if not cpdag_only:
        dag = graphs["dag"]
        validate_dag(dag)
        np.testing.assert_array_equal(dag_to_cpdag(dag), cpdag)
        report["dag_preserves_cpdag"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Five tiny updates on the selected device, checkpoint roundtrip, and graph decoding")
    parser.add_argument("--task", choices=("cpdag", "dag", "both"), default="both")
    parser.add_argument("--mechanisms", nargs="+", default=["linear_gaussian", "nonlinear_anm", "linear_nongaussian"],
                        choices=("linear_gaussian", "nonlinear_anm", "linear_nongaussian", "chebyshev"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Datasets per process/GPU; global batch = this value times world size")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--variables", type=int, default=8)
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--collider-channels", type=int, default=16)
    parser.add_argument("--data-layers", type=int, default=1)
    parser.add_argument("--cov-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--gate-init", type=float, default=.01)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--overfit-batch", action="store_true", help="Reuse one batch as an optimization sanity check")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0 for one GPU, or cuda under torchrun")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.epochs, args.steps_per_epoch = 1, 5
        args.batch_size, args.samples, args.variables = 2, 24, 5
        args.channels, args.hidden_channels, args.collider_channels, args.heads = 10, 12, 6, 2
        args.data_layers = args.cov_layers = 1
        args.mechanisms = ["nonlinear_anm"]
        args.overfit_batch = True
    if args.output_dir is None:
        args.output_dir = Path("runs/torch_smoke" if args.smoke_test else "runs/torch_bam")
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.threads) < 1:
        parser.error("epochs, steps, batch size, and threads must be positive")
    if args.samples < 2 or args.variables < 2:
        parser.error("At least two samples and two variables are required")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.task != "cpdag" and not set(args.mechanisms) & {"nonlinear_anm", "linear_nongaussian"}:
        parser.error("DAG training needs an explicitly identifiable regime; otherwise use --task cpdag")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    if world_size > 1 and int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
        parser.error("This entry point supports one server only; use torchrun --nnodes=1")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices are cpu and cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; install a CUDA-enabled PyTorch build and check driver visibility")
        if world_size > 1:
            if device.index is not None:
                parser.error("Under torchrun use --device cuda; each worker selects its LOCAL_RANK GPU")
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", device.index if device.index is not None else 0)
        if device.index >= torch.cuda.device_count():
            parser.error("Requested more GPUs than are visible to this process")
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(seconds=120))
    try:
        return _run_training(args, device, rank, world_size)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _primary_call(function, rank, distributed):
    """Run file I/O or checkpoint validation once, propagating errors to peers."""
    result = [None, None]
    if rank == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    return result[0]


def _run_training(args, device, rank, world_size):
    distributed = world_size > 1

    def prepare_output():
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory is not empty; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    _primary_call(prepare_output, rank, distributed)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model_config = dict(n_channels=args.channels, hidden_channels=args.hidden_channels,
                        collider_channels=args.collider_channels, data_layers=args.data_layers,
                        cov_layers=args.cov_layers, n_heads=args.heads, gate_init=args.gate_init,
                        include_dag_heads=args.task != "cpdag")
    model = JointBAMTorch(**model_config).to(device)
    training_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if distributed else model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_options = dict(direction_weight=0.0 if args.task == "cpdag" else 1.0,
                        order_weight=0.0 if args.task == "cpdag" else .1)
    data_seed = args.seed + rank * 1_000_003
    generator = DataGeneratorBAM(N=args.batch_size, M_min=args.samples, M_max=args.samples,
                                 d_min=args.variables, d_max=args.variables,
                                 steps_per_epoch=args.steps_per_epoch, mechanism=args.mechanisms,
                                 expected_degree=args.expected_degree, seed=data_seed)
    reference_batch = _device_batch(generator[0], device)
    with torch.no_grad():
        initial_loss = float(joint_losses(reference_batch[1], model(reference_batch[0]),
                                          distributed=distributed, **loss_options)["total"].cpu())
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    config = {"model": model.get_config(), "framework": "pytorch", "convention": "source_target",
              "loss_options": loss_options,
              "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    config["distributed"] = {"world_size": world_size, "backend": dist.get_backend() if distributed else None,
                             "batch_size_per_rank": args.batch_size,
                             "global_batch_size": args.batch_size * world_size,
                             "data_seeds": [args.seed + r * 1_000_003 for r in range(world_size)]}
    _primary_call(lambda: (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n"),
                  rank, distributed)
    if rank == 0:
        print(json.dumps({"device": str(device), **config["distributed"]}), flush=True)
    history = []
    for epoch in range(args.epochs):
        generator.set_epoch(epoch)
        reports = []
        supervised = 0
        for index in range(args.steps_per_epoch):
            observations, targets = reference_batch if args.overfit_batch else _device_batch(generator[index], device)
            reports.append(_scalars(train_step(training_model, optimizer, observations, targets, distributed=distributed, **loss_options)))
            supervised += int(targets["dag_mask"].sum().item())
        report = {name: float(np.mean([item[name] for item in reports])) for name in reports[0]}
        if distributed:
            count = torch.tensor(supervised, device=device, dtype=torch.int64)
            dist.all_reduce(count)
            supervised = int(count.item())
        report.update(epoch=epoch + 1, dag_supervised_datasets=supervised)
        history.append(report)

        def save_epoch():
            print(json.dumps(report), flush=True)
            # Save the unwrapped module: single-GPU/CPU prediction stays compatible.
            torch.save(model.state_dict(), args.output_dir / "model.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        _primary_call(save_epoch, rank, distributed)
    model.eval()
    with torch.no_grad():
        final_loss = float(joint_losses(reference_batch[1], model(reference_batch[0]),
                                        distributed=distributed, **loss_options)["total"].cpu())
    weights_changed = any(not torch.equal(before, after.detach())
                          for before, after in zip(initial_parameters, model.parameters()))
    if distributed:
        changed = torch.tensor(int(weights_changed), device=device)
        dist.all_reduce(changed, op=dist.ReduceOp.MIN)
        weights_changed = bool(changed.item())
    if not weights_changed:
        raise RuntimeError("No parameters changed during training")
    validation = {"initial_reference_loss": initial_loss, "final_reference_loss": final_loss,
                  "weights_changed": weights_changed, "finite_losses_and_gradients": True}
    validation.update(world_size=world_size, global_batch_size=args.batch_size * world_size,
                      device_type=device.type)
    if args.smoke_test:
        if distributed:
            vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            reference = vector.clone()
            dist.broadcast(reference, src=0)
            error = (vector - reference).abs().max()
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            validation["replica_max_error"] = float(error.item())
            if error.item() > 1e-6:
                raise RuntimeError("DDP model replicas differ after training")
        validation.update(_primary_call(
            lambda: _validate_checkpoint(model, args.output_dir / "model.pt", reference_batch[0],
                                         cpdag_only=args.task == "cpdag"), rank, distributed))

    def save_validation():
        (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation), flush=True)
        print(f"{'Smoke test passed' if args.smoke_test else 'Training finished'}; saved {args.output_dir}")

    _primary_call(save_validation, rank, distributed)
    return validation


if __name__ == "__main__":
    main()
