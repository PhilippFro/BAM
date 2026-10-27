"""Graph targets and structurally valid decoders for the joint BAM model.

All DAG matrices use ``A[source, target] == 1``.  A CPDAG uses the same
representation for arrows and ``C[i, j] == C[j, i] == 1`` for an undirected
edge.  Functions operate on one graph at a time and never mutate their inputs.

The CPDAG decoder first finds a high-scoring *witness DAG* on the predicted
skeleton, then converts it to its exact Markov-equivalence-class CPDAG.  This
avoids claiming that an arbitrary collection of predicted colliders forms a
valid CPDAG.  Finite-width beam search is a heuristic, not an optimality claim.
"""

from itertools import combinations
import heapq
import numpy as np


def _binary_square(matrix, name="adjacency"):
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must contain only finite binary values")
    if np.any(np.diag(a)):
        raise ValueError(f"{name} must have a zero diagonal")
    return a.astype(bool, copy=True)


def topological_order(adjacency):
    """Return a deterministic source-first order, raising on directed cycles."""
    a = _binary_square(adjacency)
    indegrees = a.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegrees == 0))
    heapq.heapify(ready)
    order = []
    while ready:
        source = int(heapq.heappop(ready))
        order.append(source)
        for target in np.flatnonzero(a[source]):
            indegrees[target] -= 1
            if indegrees[target] == 0:
                heapq.heappush(ready, int(target))
    if len(order) != len(a):
        raise ValueError("adjacency contains a directed cycle")
    return np.asarray(order, dtype=np.int64)


def validate_dag(adjacency):
    """Validate a binary DAG and return an independent uint8 copy."""
    a = _binary_square(adjacency)
    topological_order(a)
    return a.astype(np.uint8)


def skeleton_from_dag(adjacency):
    """Return the symmetric, zero-diagonal skeleton of a validated DAG."""
    a = validate_dag(adjacency).astype(bool)
    return (a | a.T).astype(np.uint8)


def collider_targets(adjacency):
    """Return ``(labels, mask)`` on axes ``[parent, center, parent]``.

    The boolean mask selects distinct-node unshielded triples in the skeleton.
    Float32 labels mark precisely ``i -> j <- k`` with no ``i-k`` edge.
    Both tensors are symmetric in their first and third axes.  When computing
    a loss, select only ``i < k`` if each triple should contribute once.
    """
    a = validate_dag(adjacency).astype(bool)
    s = a | a.T
    distinct_endpoints = ~np.eye(len(a), dtype=bool)[:, None, :]
    mask = s[:, :, None] & s.T[None, :, :] & ~s[:, None, :]
    mask &= distinct_endpoints
    labels = a[:, :, None] & a.T[None, :, :] & mask
    return labels.astype(np.float32), mask


def dag_to_cpdag(adjacency):
    """Compute the exact CPDAG using Chickering's compelled-edge labeling.

    Implements ORDER-EDGES / LABEL-EDGES, Figures 13--14 in Chickering (2002),
    *Optimal Structure Identification With Greedy Search*, JMLR 3:507--554:
    https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

    Reversible edges are returned in both directions; compelled arrows retain
    their DAG direction.  This is an exact conversion, independent of decoding
    search quality or arbitrary choices among valid topological orders.
    """
    a = validate_dag(adjacency).astype(bool)
    order = topological_order(a)
    rank = np.empty(len(a), dtype=int)
    rank[order] = np.arange(len(a))
    edges = sorted(zip(*np.nonzero(a)), key=lambda e: (rank[e[1]], -rank[e[0]]))
    # 0: no edge; 1: unknown; 2: compelled; 3: reversible.
    labels = a.astype(np.uint8)
    for x, y in edges:
        if labels[x, y] != 1:
            continue
        compelled = False
        for w in np.flatnonzero(labels[:, x] == 2):
            if not a[w, y]:
                labels[a[:, y], y] = 2
                compelled = True
                break
            labels[w, y] = 2
        if compelled:
            continue
        parents_y = np.flatnonzero(a[:, y])
        has_unshielded_parent = any(z != x and not a[z, x] for z in parents_y)
        labels[labels[:, y] == 1, y] = 2 if has_unshielded_parent else 3
    cpdag = a.astype(np.uint8)
    reversible = labels == 3
    cpdag[reversible.T] = 1
    return cpdag


def _probabilities(values, shape, name):
    p = np.asarray(values, dtype=np.float64)
    if p.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError(f"{name} must be finite probabilities in [0, 1]")
    return p.copy()


def _threshold_skeleton(skeleton_probs, threshold):
    p = np.asarray(skeleton_probs)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("skeleton_probs must be a square matrix")
    p = _probabilities(p, p.shape, "skeleton_probs")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    # Averaging makes behavior explicit for imperfectly symmetric callers.
    s = (p + p.T) / 2 > threshold
    np.fill_diagonal(s, False)
    return s


def _triples(skeleton):
    return np.asarray([
        (i, j, k)
        for j in range(len(skeleton))
        for i, k in combinations(np.flatnonzero(skeleton[:, j]), 2)
        if not skeleton[i, k]
    ], dtype=np.int64).reshape(-1, 3)


def _dag_for_order(skeleton, order):
    rank = np.empty(len(skeleton), dtype=int)
    rank[np.asarray(order, dtype=int)] = np.arange(len(skeleton))
    return (skeleton & (rank[:, None] < rank[None, :])).astype(np.uint8)


def _order_beam(skeleton, beam_width, collider_probs=None, direction_probs=None):
    """Score each edge/triple when the last of its nodes enters an order.

    Prefixes reaching the same node subset have identical future possibilities,
    so only the highest-scoring prefix for a subset needs retaining. With
    ``beam_width=None`` this is exact subset dynamic programming (exponential
    memory), permitted only for d <= 16. With finite width it is beam search.
    """
    d = len(skeleton)
    if beam_width is None:
        if d > 16:
            raise ValueError("exact decoding (beam_width=None) is limited to 16 nodes")
    elif isinstance(beam_width, (bool, np.bool_)) or int(beam_width) != beam_width or beam_width < 1:
        raise ValueError("beam_width must be a positive integer or None")
    else:
        beam_width = int(beam_width)
    epsilon = 1e-7
    triples = _triples(skeleton)
    containing = [[] for _ in range(d)]
    if collider_probs is not None:
        for i, j, k in triples:
            p = np.clip((collider_probs[i, j, k] + collider_probs[k, j, i]) / 2,
                        epsilon, 1 - epsilon)
            for v in (i, j, k):
                other = tuple(int(x) for x in (i, j, k) if x != v)
                containing[v].append(((1 << other[0]) | (1 << other[1]),
                                      float(np.log(p if v == j else 1 - p))))
    log_direction = None
    if direction_probs is not None:
        p = np.clip(direction_probs, epsilon, 1 - epsilon)
        # Conditional orientation evidence; an edge u -> v supports p[u,v]
        # and contradicts p[v,u]. This also accepts independent sigmoid heads.
        log_direction = np.log(p) + np.log1p(-p.T)
    beams = [(0.0, (), 0)]
    for _ in range(d):
        best_by_subset = {}
        for score, order, mask in beams:
            for v in range(d):
                if mask & (1 << v):
                    continue
                increment = sum(value for bits, value in containing[v]
                                if mask & bits == bits)
                if log_direction is not None:
                    increment += sum(log_direction[u, v] for u in order if skeleton[u, v])
                new_mask = mask | (1 << v)
                candidate = (float(score + increment), order + (v,), new_mask)
                old = best_by_subset.get(new_mask)
                if old is None or candidate[0] > old[0] or (
                        candidate[0] == old[0] and candidate[1] < old[1]):
                    best_by_subset[new_mask] = candidate
        beams = sorted(best_by_subset.values(), key=lambda b: (-b[0], b[1]))
        if beam_width is not None:
            beams = beams[:beam_width]
    score, order, _ = beams[0]
    return np.asarray(order, dtype=np.int64), score


def decode_cpdag(skeleton_probs, collider_probs, threshold=0.5, beam_width=32,
                 collider_threshold=0.5):
    """Decode a valid CPDAG using skeleton and collider evidence only.

    The thresholded skeleton is preserved. A source-first order is selected by
    Bernoulli collider log likelihood on unshielded triples. Its witness DAG is
    converted to a CPDAG by exact compelled-edge labeling. Incompatible collider
    requests may therefore be rejected or additional colliders introduced.

    Returns a dict with ``cpdag``, ``dag`` (the witness, NOT a causally identified
    DAG), ``order``, ``score``, ``method``, ``collider_disagreements``, and counts
    of rejected/introduced colliders relative to ``collider_threshold``.
    Finite beams provide structural validity, not an optimality guarantee.
    """
    s = _threshold_skeleton(skeleton_probs, threshold)
    c = _probabilities(collider_probs, (len(s),) * 3, "collider_probs")
    if not np.isfinite(collider_threshold) or not 0 <= collider_threshold <= 1:
        raise ValueError("collider_threshold must lie in [0, 1]")
    order, score = _order_beam(s, beam_width, collider_probs=c)
    dag = _dag_for_order(s, order)
    labels, _ = collider_targets(dag)
    rejected = introduced = 0
    for i, j, k in _triples(s):
        requested = (c[i, j, k] + c[k, j, i]) / 2 > collider_threshold
        actual = bool(labels[i, j, k])
        rejected += int(requested and not actual)
        introduced += int(actual and not requested)
    return {
        "cpdag": dag_to_cpdag(dag), "dag": dag, "order": order,
        "score": score, "method": "collider_exact" if beam_width is None else "collider_beam",
        "collider_disagreements": rejected + introduced,
        "rejected_colliders": rejected, "introduced_colliders": introduced,
    }


def _consistent_extension(cpdag, direction_probs=None, order_scores=None):
    """Score-guided Dor--Tarsi elimination, with an exact CPDAG round trip.

    A removable sink has no outgoing arrow; each undirected neighbor must be
    adjacent to all its other undirected neighbors and parents. Its parents
    need not form a clique: their already oriented colliders must be retained.
    """
    p = _binary_square(cpdag, "cpdag")
    original = p.copy()
    d = len(p)
    dag = p & ~p.T
    active = np.ones(d, dtype=bool)
    removal_order = []
    preference = None
    if direction_probs is not None:
        q = np.clip(direction_probs, 1e-7, 1 - 1e-7)
        preference = np.log(q) + np.log1p(-q.T)
    while np.any(active):
        candidates = []
        for x in np.flatnonzero(active):
            if np.any(p[x] & ~p[:, x] & active):
                continue
            neighbors = np.flatnonzero(p[x] & p[:, x] & active)
            adjacent = (p[x] | p[:, x]) & active
            valid = True
            for y in neighbors:
                others = adjacent.copy()
                others[y] = False
                if np.any(others & ~(p[y] | p[:, y])):
                    valid = False
                    break
            if not valid:
                continue
            if preference is not None:
                score = float(sum(preference[y, x] - preference[x, y] for y in neighbors))
            elif order_scores is not None:
                score = float(order_scores[x])
            else:
                score = 0.0
            order_tiebreak = 0.0 if order_scores is None else float(order_scores[x])
            candidates.append((score, order_tiebreak, int(x), neighbors))
        if not candidates:
            raise ValueError("cpdag does not admit a consistent DAG extension")
        _, _, x, neighbors = min(candidates, key=lambda item: (-item[0], -item[1], item[2]))
        dag[neighbors, x] = True
        active[x] = False
        p[x, :] = p[:, x] = False
        removal_order.append(x)
    dag = validate_dag(dag)
    if not np.array_equal(dag_to_cpdag(dag), original):
        raise ValueError("cpdag must be completed (some given arrows are not compelled)")
    score = None if preference is None else float(preference[dag.astype(bool)].sum())
    return dag, np.asarray(removal_order[::-1], dtype=np.int64), score


def decode_dag(skeleton_probs, direction_probs=None, order_scores=None,
               threshold=0.5, beam_width=32, collider_probs=None, cpdag=None):
    """Decode a fully directed DAG under the mechanism-specific model.

    If ``cpdag`` is supplied, score-guided sink elimination chooses a consistent
    extension of that CPDAG, preserving its skeleton and equivalence class.
    This greedy refinement is verified by an exact DAG-to-CPDAG round trip.
    Direction likelihood is the primary sink preference when provided;
    ``order_scores`` break ties, or select sinks alone without direction_probs.

    Otherwise, if ``order_scores`` is supplied, it defines a strict comparison mask:
    edges go from lower to higher scores, and tied-score edges are omitted.
    Otherwise beam search finds an order using pairwise direction likelihoods,
    optionally also collider likelihoods. At least one of direction_probs or
    order_scores is required, to avoid presenting an arbitrary CPDAG extension
    as a direction estimate. Unlike decode_cpdag, this function uses additional
    mechanism-specific orientation evidence. It guarantees acyclicity, not
    causal identifiability or recovery of the true DAG.
    """
    s = _threshold_skeleton(skeleton_probs, threshold)
    d = len(s)
    p = None if direction_probs is None else _probabilities(
        direction_probs, (d, d), "direction_probs")
    c = None if collider_probs is None else _probabilities(
        collider_probs, (d, d, d), "collider_probs")
    ranks = None
    if order_scores is not None:
        ranks = np.asarray(order_scores, dtype=np.float64)
        if ranks.shape != (d,) or not np.all(np.isfinite(ranks)):
            raise ValueError(f"order_scores must be a finite vector of shape {(d,)}")
    if p is None and ranks is None:
        raise ValueError("DAG decoding needs direction_probs or order_scores")
    if cpdag is not None:
        cpd = _binary_square(cpdag, "cpdag")
        if cpd.shape != s.shape or not np.array_equal(cpd | cpd.T, s):
            raise ValueError("cpdag skeleton must match the thresholded skeleton_probs")
        dag, order, score = _consistent_extension(cpd, p, ranks)
        return {"dag": dag, "order": order, "score": score,
                "method": "cpdag_consistent_greedy", "preserves_cpdag": True}
    if ranks is not None:
        dag = (s & (ranks[:, None] < ranks[None, :])).astype(np.uint8)
        return {"dag": dag, "order": np.argsort(ranks, kind="stable"),
                "score": None, "method": "strict_order_mask",
                "omitted_tied_edges": int(np.count_nonzero(s & (ranks[:, None] == ranks[None, :])) // 2)}
    order, score = _order_beam(s, beam_width, collider_probs=c, direction_probs=p)
    return {"dag": _dag_for_order(s, order), "order": order, "score": score,
            "method": "direction_exact" if beam_width is None else "direction_beam"}
