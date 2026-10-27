# PyTorch BAM: generator and training smoke test

The original repository and first joint prototype use TensorFlow. The new
`*_torch.py` implementation runs entirely in **PyTorch**, including training
and checkpoint loading. All synthetic generators, including the original
Chebyshev numerical helpers, work without TensorFlow.

## 1. Install

Extract the complete source archive and enter its `BAM2` folder first.
The folder must contain `requirements-torch.txt`. Create an environment and
install CPU PyTorch (skip environment creation if yours is already active):

```bash
python -m venv .venv-torch
source .venv-torch/bin/activate
python install_torch.py --clean-indexes
```

`--clean-indexes` bypasses inherited pip configuration and extra-index settings
for this installation only. Omit it if you need a private package mirror.
This addresses the unreachable NVIDIA index in the supplied installation log.
The helper resolves requirements relative to its own path. For manual commands
and GPU installation, see [README.md](README.md).

The CPU command was tested on Linux with Python 3.12 and PyTorch 2.14.1+cpu.
For CUDA or another platform, install the appropriate PyTorch build first,
then install the requirements. TensorFlow is not a requirement. Normal training
and prediction accept `--device cuda:0` when CUDA is available; GPU execution
has not been tested here. For single-server multi-GPU training and the container
library-path workaround, see [MULTI_GPU.md](MULTI_GPU.md).

## 2. Check the generator

```bash
python generator_torch.py
```

Its interface mirrors the original generator: `N` is the number of datasets in
one training batch, `M_min/M_max` bound the observations per dataset, and
`d_min/d_max` bound the number of variables. One M and d are sampled per batch.

```python
from generator_torch import DataGeneratorBAM

generator = DataGeneratorBAM(
    N=2,
    M_min=32, M_max=64,
    d_min=4, d_max=8,
    steps_per_epoch=20,
    mechanism="nonlinear_anm",
    seed=42,
)

X, labels = generator[0]
print(X.shape)                     # [N, observations, variables]
print(labels["skeleton"].shape)    # [N, variables, variables]
print(labels["colliders"].shape)   # [N, parent, center, parent]
print(labels["dag_mask"])          # which datasets permit DAG supervision
```

The returned values are CPU tensors. `generator.set_epoch(epoch)` produces a
fresh, reproducible stream each epoch. If using `DataLoader`, set
`batch_size=None`, because each generator item is already a complete batch.

Supported mechanisms:

| Name | Data | Default DAG supervision |
|---|---|---|
| `nonlinear_anm` | Smooth nonlinear functions with independent additive Gaussian noise | Enabled under these generating assumptions |
| `linear_nongaussian` | Linear functions with independent Laplace noise | Enabled under these generating assumptions |
| `linear_gaussian` | Linear functions with varying Gaussian noise variances | Disabled |
| `chebyshev` | Original Chebyshev generator's numerical functions | Disabled unless assumptions are explicitly overridden |

Pass a list of names to mix regimes per graph. All regimes supervise skeleton
and collider predictions. `dag_mask` disables **both** direction and order
losses for ambiguous regimes; knowing a simulated DAG does not itself establish
observational identifiability. Chebyshev helper bodies are unchanged from the
original code, extracted into `chebyshev_numpy.py` to remove its TensorFlow
import. Tests verify identical draws for identical legacy NumPy seeds.

## 3. Run the smoke test

```bash
python train_joint_torch.py --smoke-test
```

It runs five small updates using one nonlinear-ANM batch per worker and checks
the following. The default is CPU; add `--device cuda` to test a GPU:

- Finite forward losses and backward gradients.
- Actual parameter updates.
- Saving and reloading the checkpoint with matching predictions.
- Skeleton → CPDAG → DAG decoding, with `dag_to_cpdag(dag) == cpdag`.

The tested default seed reduced the fixed-batch loss from **2.19287 to 2.18260**.
Checkpoint predictions matched exactly. This checks execution and optimization;
five updates do not produce a useful causal-discovery model. The script reports
the loss change but does not require every seed to improve after five steps.

Outputs go to `runs/torch_smoke/`:

| File | Contents |
|---|---|
| `config.json` | Model, training options, and graph convention |
| `model.pt` | PyTorch `state_dict`, with no serialized model object |
| `history.json` | Per-epoch loss components |
| `validation.json` | Update, checkpoint, and decoding checks |

Existing output directories are not overwritten. To rerun:

```bash
python train_joint_torch.py --smoke-test --output-dir runs/torch_smoke_2
```

## 4. Train on fresh batches

```bash
python train_joint_torch.py \
  --task both \
  --mechanisms nonlinear_anm linear_nongaussian linear_gaussian \
  --epochs 5 --steps-per-epoch 100 \
  --batch-size 4 --samples 64 --variables 8 \
  --output-dir runs/torch_bam
```

This is a starting example, not a tuned training schedule. The CLI uses fixed
sample and variable counts; the generator API supports ranges. Use
`--overfit-batch` to repeat one batch for an optimization sanity check.

For CPDAG-only training, including the original Chebyshev data:

```bash
python train_joint_torch.py --task cpdag --mechanisms chebyshev \
  --epochs 2 --steps-per-epoch 100 --output-dir runs/torch_cpdag
```

`--task cpdag` omits the direction and ordering heads. `--task dag` still trains
the skeleton and collider heads, because DAG inference refines a CPDAG.

A minimal custom update, using the variable-sized generator above:

```python
import torch
from joint_model_torch import JointBAMTorch
from joint_training_torch import train_step

torch.set_num_threads(1)  # convenient for these small CPU batches
model = JointBAMTorch(n_channels=20, n_heads=5, data_layers=1, cov_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(2):
    generator.set_epoch(epoch)
    for index in range(len(generator)):
        X, labels = generator[index]
        losses = train_step(model, optimizer, X, labels)
    print(epoch, float(losses["total"]))
```

## 5. Predict

```bash
python predict_joint_torch.py --model-dir runs/torch_bam \
  --input observations.npy --output predictions.npz --target both
```

`observations.npy` must contain one `[samples,variables]` matrix. An `.npz` file
with key `X` is also accepted. Prediction standardizes columns as in training.
Use `--target cpdag` to stop before the mechanism-dependent refinement.

```python
from predict_joint_torch import load_model, predict

model = load_model("runs/torch_bam")
graphs, scores = predict(model, observations, target="both")
skeleton, cpdag, dag = graphs["skeleton"], graphs["cpdag"], graphs["dag"]
```

One network pass produces all scores. The CPDAG decoder uses skeleton/collider
evidence only; the DAG decoder uses conditional direction preferences, with
ordering as a tie-breaker. Refinement preserves the CPDAG's equivalence class,
including any errors already made in its skeleton or colliders. A DAG estimate
does not certify that the observed data identify all its arrows.

All new matrices use **`A[source,target]`**, the transpose of the original
generator's convention. In the CPDAG matrix, entries in both directions mean
an undirected edge. See [JOINT_MODEL.md](JOINT_MODEL.md) for decoding details.

## Validation and port details

```bash
python -m pytest -q tests
```

The updated PyTorch-only environment passed **68 tests**. Two TensorFlow-specific
modules and three opt-in distributed integration cases were skipped. The latter
need a host permitting local communication sockets; see MULTI_GPU.md. Tests include
actual forward/backward passes, nonzero gradients in both feature branches,
sample/variable permutation properties, dynamic sizes, constant inputs,
matrix-log gradient checks, checkpoint round trips, generators, and graph
consistency. A separate mixed-regime training run also completed with fresh
batches and masked DAG supervision.

This is an architectural port, not a TensorFlow checkpoint conversion. Positive
softplus parameters replace projected nonnegative BAM weights, and softmax
coefficients replace projected polynomial coefficients. The matrix logarithm
uses a custom first-order derivative with stable repeated-eigenvalue limits;
higher-order derivatives are unsupported. These choices preserve the intended
SPD operations but do not promise numerical parity with TensorFlow training.

The dense collider head has cubic memory growth in variable count. Start with
small graphs. There are no causal-recovery benchmarks or calibration claims;
CPDAG/DAG decoding guarantees graph structure consistency, not causal accuracy.
