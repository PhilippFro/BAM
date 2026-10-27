"""PyTorch objectives and one optimizer step for the joint BAM model.

Arrays use A[source, target]. Mechanism-dependent supervision is enabled by
an explicit per-dataset ``dag_mask``, not by a model-confidence threshold.
"""

import math

import torch
import torch.distributed as dist


def _masked_graph_mean(values, mask, distributed=False):
    """Weight eligible graphs equally, counting each graph's entries once."""
    mask = torch.broadcast_to(mask.to(values), values.shape)
    axes = tuple(range(1, values.ndim))
    counts = mask.sum(dim=axes)
    per_graph = (values * mask).sum(dim=axes) / counts.clamp_min(1)
    eligible = (counts > 0).to(values)
    numerator = (per_graph * eligible).sum()
    count = eligible.sum()
    if not distributed:
        return numerator / count.clamp_min(1)
    # DDP averages gradients across ranks. Scale the differentiable local sum
    # by world_size so the result equals a single combined-batch objective,
    # including when a rank has no eligible colliders or DAG-supervised graphs.
    stats = torch.stack((numerator.detach(), count.detach()))
    dist.all_reduce(stats)
    denominator = stats[1].clamp_min(1)
    local = numerator * dist.get_world_size() / denominator
    # Report the same global scalar on every rank without differentiating the
    # all_reduce or changing the local gradient that DDP will average.
    return local + (stats[0] / denominator - local.detach())


def _binary_crossentropy(labels, probabilities):
    probabilities = probabilities.clamp(1e-7, 1 - 1e-7)
    return -(labels * probabilities.log() + (1 - labels) * torch.log1p(-probabilities))


def joint_losses(targets, predictions, collider_weight=1.0,
                 direction_weight=1.0, order_weight=0.1, order_margin=1.0,
                 distributed=False):
    """Return total, skeleton, collider, direction, and order losses.

    Skeleton and direction BCE count unordered pairs once. Collider BCE counts
    each pair of endpoints once and uses only true unshielded triples. Direction
    BCE is conditional on a true skeleton edge; order hinge loss constrains only
    true parent-child pairs. Both DAG losses are masked by ``dag_mask``. Empty
    eligible sets contribute differentiable zero rather than NaN.
    ``distributed=True`` requires an initialized process group, identical loss
    options on all ranks, and DDP gradient averaging for correct normalization.
    """
    weights = (collider_weight, direction_weight, order_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Loss weights must be finite and nonnegative")
    if not math.isfinite(order_margin) or order_margin <= 0:
        raise ValueError("order_margin must be finite and positive")
    reference = predictions["skeleton"]
    if reference.ndim != 3 or reference.shape[-1] != reference.shape[-2]:
        raise ValueError("skeleton predictions must have shape [batch,d,d]")
    target = {name: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
              for name, value in targets.items()}
    skeleton = target["skeleton"]
    d = skeleton.shape[-1]
    upper = torch.ones((d, d), device=reference.device, dtype=reference.dtype).triu(1)[None]
    skeleton_loss = _masked_graph_mean(_binary_crossentropy(skeleton, reference), upper, distributed)
    collider_mask = target["collider_mask"] * upper[:, :, None, :]
    collider_loss = _masked_graph_mean(
        _binary_crossentropy(target["colliders"], predictions["colliders"]), collider_mask, distributed)
    direction_loss = reference.sum() * 0
    order_loss = reference.sum() * 0
    if direction_weight > 0 or order_weight > 0:
        dag = target["dag"]
        mask = target["dag_mask"][:, None, None]
        if not torch.all(torch.isfinite(mask) & ((mask == 0) | (mask == 1))):
            raise ValueError("dag_mask must contain only finite binary values")
        if direction_weight > 0:
            direction_loss = _masked_graph_mean(
                _binary_crossentropy(dag, predictions["direction"]), upper * skeleton * mask, distributed)
        if order_weight > 0:
            ranks = predictions["order"]
            violations = torch.relu(order_margin + ranks[:, :, None] - ranks[:, None, :])
            order_loss = _masked_graph_mean(violations, dag * mask, distributed)
    total = (skeleton_loss + collider_weight * collider_loss
             + direction_weight * direction_loss + order_weight * order_loss)
    return {"total": total, "skeleton": skeleton_loss, "colliders": collider_loss,
            "direction": direction_loss, "order": order_loss}


def train_step(model, optimizer, observations, targets, max_grad_norm=1.0, **loss_options):
    """Run a checked optimizer update and return detached scalar losses."""
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(observations)
    losses = joint_losses(targets, predictions, **loss_options)
    if not torch.stack([torch.isfinite(value) for value in losses.values()]).all().item():
        raise FloatingPointError("Non-finite joint loss")
    losses["total"].backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("No model parameters received gradients")
    # One host synchronization per check rather than one per parameter on CUDA.
    if not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item():
        raise FloatingPointError("Non-finite model gradient")
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item():
        raise FloatingPointError("Optimizer produced non-finite model parameters")
    return {name: value.detach() for name, value in losses.items()}
