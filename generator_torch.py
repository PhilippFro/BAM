"""Simple PyTorch batches with the original BAM generator's N/M/d interface.

Each dataset item is already a complete batch of N simulated datasets. Use
``DataLoader(generator, batch_size=None)`` so PyTorch does not add a batch axis.
The generator has no TensorFlow dependency, including its Chebyshev option.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from joint_data_BAM import MECHANISMS, sample_batch


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class DataGeneratorBAM(Dataset):
    """Generate CPU tensor batches for one shared CPDAG/DAG model.

    Parameters resemble the original ``DataGeneratorChebyshev``: ``N`` is the
    number of datasets/graphs in each training batch, ``M_min/M_max`` bound the
    observations per dataset, and ``d_min/d_max`` bound the variable count.
    One M and d are sampled per batch, making its tensors rectangular.

    ``mechanism`` accepts a supported regime or a sequence of regimes sampled
    per graph. DAG supervision is explicitly masked by the generating regime;
    it is off for unrestricted linear-Gaussian and legacy Chebyshev samples.

    Samples are deterministic for (seed, epoch, index), independent of access
    order. Call ``set_epoch`` before each epoch to obtain fresh data. With a
    DataLoader using workers, recreate its iterator after setting the epoch and
    use ``persistent_workers=False`` so workers see the updated epoch value.
    Move returned tensors to a GPU in the training loop, not in this generator.
    """

    def __init__(self, N=2, M_min=64, M_max=128, d_min=4, d_max=12,
                 steps_per_epoch=128, mechanism="nonlinear_anm",
                 expected_degree=2.0, seed=0, standardize=True,
                 dag_supervision=None):
        self.N = _integer(N, "N", 1)
        self.M_min = _integer(M_min, "M_min", 2)
        self.M_max = _integer(M_max, "M_max", self.M_min)
        self.d_min = _integer(d_min, "d_min", 1)
        self.d_max = _integer(d_max, "d_max", self.d_min)
        self.steps_per_epoch = _integer(steps_per_epoch, "steps_per_epoch", 1)
        self.seed = _integer(seed, "seed")
        self.epoch = 0
        self.mechanism = (mechanism,) if isinstance(mechanism, str) else tuple(mechanism)
        if not self.mechanism or any(name not in MECHANISMS for name in self.mechanism):
            raise ValueError(f"mechanism must contain names from {MECHANISMS}")
        if not np.isscalar(expected_degree) or not np.isfinite(expected_degree) or expected_degree < 0:
            raise ValueError("expected_degree must be finite and nonnegative")
        if not isinstance(standardize, (bool, np.bool_)):
            raise ValueError("standardize must be a boolean")
        if dag_supervision is not None and not isinstance(dag_supervision, (bool, np.bool_)):
            raise ValueError("dag_supervision must be None or a boolean")
        self.expected_degree = float(expected_degree)
        self.standardize = bool(standardize)
        self.dag_supervision = dag_supervision

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        """Change the deterministic stream for the next training epoch."""
        self.epoch = _integer(epoch, "epoch")

    def __getitem__(self, index):
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(self)):
            raise IndexError("batch index is outside this epoch")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        samples = int(rng.integers(self.M_min, self.M_max + 1))
        variables = int(rng.integers(self.d_min, self.d_max + 1))
        observations, targets = sample_batch(
            self.N, samples, variables, mechanism=self.mechanism,
            expected_degree=self.expected_degree, rng=rng,
            standardize=self.standardize, dag_supervision=self.dag_supervision,
        )
        return torch.from_numpy(observations), {
            key: torch.from_numpy(value) for key, value in targets.items()
        }


if __name__ == "__main__":
    generator = DataGeneratorBAM(N=2, M_min=16, M_max=24, d_min=4, d_max=6)
    x, labels = generator[0]
    print("Observations:", tuple(x.shape))
    print("Targets:", {key: tuple(value.shape) for key, value in labels.items()})
    print("DAG supervision mask:", labels["dag_mask"].tolist())
