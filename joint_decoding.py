"""Framework-independent skeleton, CPDAG, and DAG decoding."""

import numpy as np

from graph_utils_BAM import decode_cpdag, decode_dag


def decode_predictions(predictions, target="both", threshold=0.5, beam_width=32):
    """Decode one dataset's neural scores, without re-running the network.

    The CPDAG stage uses only skeleton and collider predictions. The DAG stage
    refines that CPDAG using mechanism scores, preserving its equivalence class.
    ``target='dag'`` still runs the CPDAG stage. No identifiability decision is
    inferred from score magnitudes. The returned DAG is a model-dependent
    estimate; the CPDAG stage's internal witness is deliberately not exposed as
    an additional causal prediction.
    """
    if target not in {"skeleton", "cpdag", "dag", "both"}:
        raise ValueError("target must be skeleton, cpdag, dag, or both")
    p = np.asarray(predictions["skeleton"], dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 1:
        raise ValueError("skeleton predictions must be a nonempty square matrix")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("skeleton probabilities must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    skeleton = ((p + p.T) / 2 > threshold).astype(np.int8)
    np.fill_diagonal(skeleton, 0)
    result = {"skeleton": skeleton, "metadata": {"adjacency_convention": "source_target"}}
    if target == "skeleton":
        return result
    cpdag_result = decode_cpdag(p, predictions["colliders"], threshold=threshold,
                                beam_width=beam_width)
    result["cpdag"] = cpdag_result["cpdag"]
    result["metadata"].update({"cpdag_method": cpdag_result["method"],
                                "collider_disagreements": cpdag_result["collider_disagreements"]})
    if target in {"dag", "both"}:
        if "direction" not in predictions:
            raise ValueError("DAG refinement requires mechanism-dependent direction predictions; "
                             "use a model trained with DAG supervision or request target='cpdag'")
        refined = decode_dag(p, direction_probs=predictions["direction"],
                             order_scores=predictions.get("order"), threshold=threshold,
                             beam_width=beam_width, cpdag=result["cpdag"])
        result["dag"] = refined["dag"]
        result["order"] = np.asarray(refined["order"], dtype=np.int64)
        result["metadata"]["dag_method"] = refined["method"]
        result["metadata"]["dag_interpretation"] = "mechanism_dependent_point_estimate"
    return result
