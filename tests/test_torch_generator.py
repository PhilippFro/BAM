"""Generator contracts, including the original Chebyshev numerical semantics."""

import ast
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import chebyshev_numpy
from generator_torch import DataGeneratorBAM
from joint_data_BAM import sample_batch


@pytest.mark.parametrize("mechanism, mask", [
    ("linear_gaussian", 0), ("linear_nongaussian", 1),
    ("nonlinear_anm", 1), ("chebyshev", 0),
])
def test_all_mechanisms_produce_finite_cpu_batches(mechanism, mask):
    generator = DataGeneratorBAM(N=2, M_min=17, M_max=17, d_min=5, d_max=5,
                                 mechanism=mechanism, seed=7)
    x, labels = generator[0]
    assert x.shape == (2, 17, 5)
    assert x.dtype == torch.float32 and x.device.type == "cpu"
    assert torch.isfinite(x).all()
    for value in labels.values():
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        assert torch.isfinite(value).all()
    assert labels["dag_mask"].tolist() == [mask, mask]
    assert labels["colliders"].shape == (2, 5, 5, 5)
    assert torch.equal(labels["skeleton"], labels["skeleton"].transpose(1, 2))


def test_access_order_reproducibility_and_new_epochs():
    generator = DataGeneratorBAM(N=1, M_min=10, M_max=10, d_min=4, d_max=4, seed=9)
    first_x, first_labels = generator[3]
    generator[4]
    again_x, again_labels = generator[3]
    assert torch.equal(first_x, again_x)
    for key in first_labels:
        assert torch.equal(first_labels[key], again_labels[key])
    generator.set_epoch(1)
    next_x, _ = generator[3]
    assert not torch.equal(first_x, next_x)
    generator.set_epoch(0)
    assert torch.equal(first_x, generator[3][0])


def test_loader_receives_whole_batches_with_variable_shapes():
    generator = DataGeneratorBAM(N=2, M_min=8, M_max=13, d_min=3, d_max=6,
                                 steps_per_epoch=8, seed=10)
    shapes = set()
    for x, labels in DataLoader(generator, batch_size=None, num_workers=0):
        assert x.ndim == 3 and x.shape[0] == 2
        assert 8 <= x.shape[1] <= 13 and 3 <= x.shape[2] <= 6
        assert labels["dag"].shape == (2, x.shape[2], x.shape[2])
        shapes.add(tuple(x.shape))
    assert len(shapes) > 1


def test_tensor_labels_exactly_match_numpy_generator():
    seed, epoch, index = 71, 3, 2
    generator = DataGeneratorBAM(N=2, M_min=8, M_max=13, d_min=3, d_max=6, seed=seed)
    generator.set_epoch(epoch)
    x, labels = generator[index]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    m, d = int(rng.integers(8, 14)), int(rng.integers(3, 7))
    expected_x, expected_labels = sample_batch(2, m, d, rng=rng)
    np.testing.assert_array_equal(x.numpy(), expected_x)
    for key in labels:
        np.testing.assert_array_equal(labels[key].numpy(), expected_labels[key])


def test_chebyshev_helpers_preserve_original_numerical_semantics():
    # Compile only the original's NumPy function definitions; never import TF.
    original = Path(__file__).resolve().parents[1] / "generator_cheby_BAM.py"
    tree = ast.parse(original.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"np": np, "math": math}
    exec(compile(tree, str(original), "exec"), namespace)
    dag_child_parent = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    state = np.random.get_state()
    try:
        np.random.seed(67)
        reference = namespace["sample_from_dag_chebyshev"](dag_child_parent, 20)
        np.random.seed(67)
        actual = chebyshev_numpy.sample_from_dag_chebyshev(dag_child_parent, 20)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(actual, reference)


def test_all_generators_work_when_tensorflow_import_is_forbidden():
    code = """
import sys
class BlockTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tensorflow' or fullname.startswith('tensorflow.'):
            raise AssertionError('Unexpected TensorFlow dependency: ' + fullname)
sys.meta_path.insert(0, BlockTensorFlow())
from generator_torch import DataGeneratorBAM
for name in ('linear_gaussian', 'linear_nongaussian', 'nonlinear_anm', 'chebyshev'):
    x, labels = DataGeneratorBAM(N=1,M_min=8,M_max=8,d_min=3,d_max=3,mechanism=name)[0]
    assert x.shape == (1,8,3)
assert 'tensorflow' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)


def test_invalid_arguments_and_indices():
    for kwargs in ({"N": 0}, {"M_min": 1}, {"M_min": 8, "M_max": 7},
                   {"d_min": 5, "d_max": 3}, {"seed": -1}, {"seed": True},
                   {"mechanism": "unknown"}, {"dag_supervision": .5}):
        with pytest.raises(ValueError):
            DataGeneratorBAM(**kwargs)
    generator = DataGeneratorBAM(steps_per_epoch=2)
    for index in (-1, 2, True, .5):
        with pytest.raises(IndexError):
            generator[index]
    with pytest.raises(ValueError):
        generator.set_epoch(-1)
