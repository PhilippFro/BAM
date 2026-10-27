"""Simulation and labels for the joint CPDAG / DAG BAM model.

All new APIs use ``dag[source, target] == 1``.  Collider tensors have axes
``[parent, center, parent]`` and are symmetric in the first and last axes.

``dag_mask`` is a declaration about the simulation regime, not an estimated
confidence or a test of identifiability.  CPDAG supervision is always enabled.
The built-in linear non-Gaussian and nonlinear additive-noise regimes permit
DAG supervision under their independent-noise, no-hidden-confounder assumptions.
The unrestricted linear-Gaussian and legacy Chebyshev regimes default to CPDAG
supervision only.  Gaussian *noise* does not preclude nonlinear identification.
"""

from contextlib import contextmanager
from threading import RLock

import numpy as np

from graph_utils_BAM import collider_targets


MECHANISMS = (
    "linear_gaussian",
    "linear_nongaussian",
    "nonlinear_anm",
    "chebyshev",
)
DAG_SUPERVISED_MECHANISMS = frozenset(("linear_nongaussian", "nonlinear_anm"))
_LEGACY_RNG_LOCK = RLock()


def make_targets(dag, dag_supervision=False):
    """Build one graph's labels without mutating the source adjacency.

    Parameters
    ----------
    dag : array-like, shape (d, d)
        Binary DAG with the source-to-target convention.
    dag_supervision : bool
        Whether directed labels should contribute to the training loss.  Use
        ``False`` when the generating assumptions do not justify selecting a
        particular member of the Markov-equivalence class.

    Returns
    -------
    dict
        Float32 arrays named ``skeleton``, ``colliders``, ``collider_mask``,
        ``dag``, and a scalar ``dag_mask``.  ``collider_mask`` selects all
        unshielded triples in the true skeleton, including negative examples.
        Directed labels are retained even when masked, for explicit evaluation.
    """
    if not isinstance(dag_supervision, (bool, np.bool_)):
        raise ValueError("dag_supervision must be a boolean")
    colliders, mask = collider_targets(dag)
    adjacency = np.array(dag, dtype=np.float32, copy=True)
    return {
        "skeleton": np.maximum(adjacency, adjacency.T),
        "colliders": np.asarray(colliders, dtype=np.float32),
        "collider_mask": np.asarray(mask, dtype=np.float32),
        "dag": adjacency,
        "dag_mask": np.float32(dag_supervision),
    }


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _generator(seed=None, rng=None):
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def _mechanism_choices(mechanism):
    choices = (mechanism,) if isinstance(mechanism, str) else tuple(mechanism)
    if not choices or any(name not in MECHANISMS for name in choices):
        raise ValueError(f"mechanism must contain names from {MECHANISMS}")
    return choices


def _sample_adjacency(d, expected_degree, rng):
    """Sample an ordered ER skeleton; expected_degree is its mean degree."""
    probability = min(expected_degree / max(d - 1, 1), 1.0)
    return np.triu(rng.random((d, d)) < probability, k=1).astype(np.float32)


@contextmanager
def _legacy_random_seed(seed):
    """Confine legacy NumPy-global RNG use and restore the caller's state.

    The lock serializes calls made through this wrapper; unrelated code using
    NumPy's legacy global RNG should not run concurrently with Chebyshev draws.
    The three built-in mechanisms never touch the global RNG.
    """
    with _LEGACY_RNG_LOCK:
        previous = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(previous)


def _sample_observations(dag, num_samples, mechanism, rng):
    if mechanism == "chebyshev":
        # Numerical helpers are copied unchanged into a NumPy-only module so
        # this legacy mechanism also works without a TensorFlow installation.
        from chebyshev_numpy import sample_from_dag_chebyshev

        legacy_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        with _legacy_random_seed(legacy_seed):
            # Legacy code expects child,parent and topologically ordered nodes.
            return sample_from_dag_chebyshev(dag.T.copy(), num_samples)

    d = len(dag)
    samples = np.zeros((num_samples, d), dtype=np.float64)
    noise_scales = rng.uniform(0.5, 1.5, size=d)
    if mechanism == "linear_nongaussian":
        noise = rng.laplace(size=(num_samples, d)) / np.sqrt(2.0)
    else:
        noise = rng.normal(size=(num_samples, d))
    noise *= noise_scales

    for child in range(d):
        parents = np.flatnonzero(dag[:, child])
        samples[:, child] = noise[:, child]
        if parents.size:
            coefficients = rng.uniform(0.5, 1.5, size=len(parents))
            coefficients *= rng.choice((-1.0, 1.0), size=len(parents))
            parent_values = samples[:, parents]
            if mechanism == "nonlinear_anm":
                # Smooth nonlinearity in every parent, independent additive
                # Gaussian noise, and no latent common causes.  These are
                # explicit modeling restrictions, not a blanket claim that
                # arbitrary nonlinear systems are identifiable.
                slopes = rng.uniform(0.5, 1.5, size=len(parents))
                parent_values = np.tanh(parent_values * slopes)
            samples[:, child] += parent_values @ coefficients
    return samples


def sample_batch(
    batch_size,
    num_samples,
    num_variables,
    mechanism="nonlinear_anm",
    expected_degree=2.0,
    seed=None,
    rng=None,
    standardize=True,
    dag_supervision=None,
):
    """Sample observations and joint labels using one graph per batch item.

    ``mechanism`` may be a name or a nonempty sequence of names.  For a sequence,
    a regime is sampled uniformly for each graph, and its own DAG supervision
    mask is used.  ``expected_degree`` is the expected undirected skeleton
    degree before saturation at a complete graph.  Graph labels and observations
    receive the same random variable permutation after simulation.

    Built-in mechanisms use only a local ``numpy.random.Generator``.  The
    optional Chebyshev adapter seeds/restores the legacy global RNG.  A boolean
    ``dag_supervision`` explicitly overrides regime defaults for every item;
    use it only when the caller intends those additional assumptions.

    Returns ``X`` of shape ``(batch_size, num_samples, num_variables)`` and a
    dictionary of stacked targets.  All returned arrays are float32.
    """
    batch_size = _positive_integer(batch_size, "batch_size")
    num_samples = _positive_integer(num_samples, "num_samples", minimum=2)
    num_variables = _positive_integer(num_variables, "num_variables")
    if not np.isscalar(expected_degree) or not np.isfinite(expected_degree) or expected_degree < 0:
        raise ValueError("expected_degree must be finite and nonnegative")
    if dag_supervision is not None and not isinstance(dag_supervision, (bool, np.bool_)):
        raise ValueError("dag_supervision must be None or a boolean")
    choices = _mechanism_choices(mechanism)
    rng = _generator(seed, rng)
    observations, labels = [], []

    for _ in range(batch_size):
        selected = choices[int(rng.integers(len(choices)))]
        dag = _sample_adjacency(num_variables, float(expected_degree), rng)
        values = _sample_observations(dag, num_samples, selected, rng)
        if standardize:
            values = values - values.mean(axis=0, keepdims=True)
            values = values / np.maximum(values.std(axis=0, keepdims=True), 1e-8)
        permutation = rng.permutation(num_variables)
        values = values[:, permutation]
        dag = dag[np.ix_(permutation, permutation)]
        supervise = selected in DAG_SUPERVISED_MECHANISMS if dag_supervision is None else dag_supervision
        observations.append(np.asarray(values, dtype=np.float32))
        labels.append(make_targets(dag, dag_supervision=supervise))

    return np.stack(observations), {
        key: np.stack([item[key] for item in labels]) for key in labels[0]
    }


class JointDataGenerator:
    """Small NumPy batch stream for custom training loops.

    Sample and variable counts may each be an integer or an inclusive
    ``(minimum, maximum)`` pair.  A batch uses one shared shape.  Indices select
    training steps rather than cached batches: repeated access produces fresh
    draws.  Equal seeds and equal call sequences reproduce the same stream.
    """

    def __init__(
        self,
        batch_size=1,
        num_samples=128,
        num_variables=10,
        mechanism="nonlinear_anm",
        expected_degree=2.0,
        steps_per_epoch=128,
        seed=None,
        standardize=True,
        dag_supervision=None,
    ):
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.num_samples = self._range(num_samples, "num_samples", minimum=2)
        self.num_variables = self._range(num_variables, "num_variables")
        self.mechanism = _mechanism_choices(mechanism)
        self.expected_degree = expected_degree
        self.steps_per_epoch = _positive_integer(steps_per_epoch, "steps_per_epoch")
        self.rng = _generator(seed)
        self.standardize = standardize
        self.dag_supervision = dag_supervision

    @staticmethod
    def _range(value, name, minimum=1):
        if isinstance(value, (int, np.integer)):
            integer = _positive_integer(value, name, minimum)
            return integer, integer
        try:
            low, high = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer or (minimum, maximum)") from exc
        low = _positive_integer(low, name, minimum)
        high = _positive_integer(high, name, minimum)
        if low > high:
            raise ValueError(f"{name} minimum must not exceed maximum")
        return low, high

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError("batch index is outside this epoch")
        sample_count = int(self.rng.integers(self.num_samples[0], self.num_samples[1] + 1))
        variable_count = int(self.rng.integers(self.num_variables[0], self.num_variables[1] + 1))
        return sample_batch(
            self.batch_size,
            sample_count,
            variable_count,
            mechanism=self.mechanism,
            expected_degree=self.expected_degree,
            rng=self.rng,
            standardize=self.standardize,
            dag_supervision=self.dag_supervision,
        )
