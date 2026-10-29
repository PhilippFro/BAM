# BAM2: joint causal graph estimation in PyTorch

BAM2 is a complete source repository; It extends [PhilippFro/BAM](https://github.com/PhilippFro/BAM)
with one PyTorch network for a single model **skeleton → CPDAG → DAG** estimation. It includes
the model, synthetic data generators, training and prediction scripts, requirements,
and tests. Code was updated from an old Tensorflow version to an actual Pytorch version.

A directional stream was added to the BAM stream controlled
by learnable channel gates. Skeleton, collider, direction,
and ordering heads share the network. Three structured decoding stages follow
one neural forward pass.

## Start here

Extract `BAM2.zip`, then enter its `BAM2` folder. Run these commands
**inside that folder**, where `requirements-torch.txt` and `train_joint_torch.py`
are located. If you already have an activated virtual environment, skip the
first two commands.

```bash
python -m venv .venv-torch
source .venv-torch/bin/activate
python install_torch.py --clean-indexes
python train_joint_torch.py --smoke-test
```

On Windows PowerShell, activate with `.venv-torch\Scripts\Activate.ps1` instead.
Python 3.12 on Linux is the tested environment. The installer uses the Python
environment that runs it; it does not create or activate an environment itself.
It resolves the requirements file relative to its own location, so it can also
be invoked by absolute path from another directory.

`--clean-indexes` ignores inherited pip configuration and package-index environment
settings for the installer subprocesses. This bypasses stale extra indexes such
as an unreachable `pypi.ngc.nvidia.com`; it does not change your saved configuration.
Omit this flag if your organization requires its own package mirror or pip settings.
The behavior follows [pip's configuration documentation](https://pip.pypa.io/en/stable/topics/configuration/).

Manual installation is also supported:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-torch.txt
```

For GPU training, first install the appropriate CUDA build using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then run
`python install_torch.py --skip-torch` and train with `--device cuda:0`.
GPU execution has not been validated here. The smoke test defaults to CPU and
honors an explicit `--device cuda`.

## Single-server multi-GPU training

The training script now supports one worker per GPU with DistributedDataParallel.
See [MULTI_GPU.md](MULTI_GPU.md) for CUDA installation, the container-compatible
launcher, and commands for 4 or 8 GPUs. With your intended virtual environment
activated (a new empty environment is also supported):

```bash
python -E -s run_bam.py install_torch.py --clean-indexes --cuda
python -E -s run_bam.py -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --local-addr=127.0.0.1 \
  train_joint_torch.py --device cuda --smoke-test --output-dir runs/ddp_smoke_4gpu
```

The CUDA installer selects `torch==2.14.1+cu130`; see the guide for driver
requirements. `--batch-size` is per GPU. Checkpoints remain compatible with
CPU prediction. Distributed loss normalization accounts for varying numbers of
eligible collider/direction targets on each worker. Local loss/gradient checks
pass, but real DDP execution is unverified here because communication sockets
are restricted and CUDA is unavailable. The server smoke test checks it.

## Troubleshooting: native-library conflict in a container

If Python 3.14 loads the virtual environment's `torch/__init__.py` but the error
names `/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_python.so`,
two different PyTorch installations are being mixed. A missing CPython symbol
such as `_PyThreadState_UncheckedGet` is consistent with this binary mismatch.
The log alone does not identify which loader setting introduced the old library.
`pip check` only checks package metadata, so the installer now also imports
PyTorch and runs a small CPU backward pass before reporting success.

With the virtual environment activated, try a CPU run with inherited library
and Python search overrides removed **for this command only**:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME \
  "$VIRTUAL_ENV/bin/python" -E -s train_joint_torch.py --smoke-test
```

`-E -s` ignores Python environment overrides and user site packages while keeping
the script directory available for the model's local imports. Unlike `-I`, it
does not remove that directory. These commands do not modify the container,
delete its system PyTorch, or change your shell's environment.

If it succeeds, use the same prefix for subsequent CPU training commands.
If a prior run populated `runs/torch_smoke`, add a fresh `--output-dir`.
Clearing all library overrides is a CPU diagnostic; it is not a general CUDA
setup recipe, since GPU/container libraries can depend on those paths.

If it still fails, collect these diagnostics (they do not import PyTorch):

```bash
python -E -s -c 'import sys, importlib.util; print(sys.executable); print(sys.version); print(importlib.util.find_spec("torch").origin)'
printf 'LD_LIBRARY_PATH=%s\nLD_PRELOAD=%s\nPYTHONPATH=%s\n' "$LD_LIBRARY_PATH" "$LD_PRELOAD" "$PYTHONPATH"
```

Keep the resulting traceback as well. A remaining conflict may require inspecting
the native loader configuration rather than reinstalling the same wheel.
References: [Linux dynamic loader](https://man7.org/linux/man-pages/man8/ld.so.8.html)
and [Python command-line options](https://docs.python.org/3.14/using/cmdline.html).

## Check and train

```bash
# Generate one batch and display its tensor shapes.
python generator_torch.py

# Five small updates, gradient checks, checkpoint reload, and graph decoding.
python train_joint_torch.py --smoke-test

# Train on newly generated datasets with mixed mechanisms.
python train_joint_torch.py --task both \
  --mechanisms nonlinear_anm linear_nongaussian linear_gaussian \
  --epochs 5 --steps-per-epoch 100 \
  --batch-size 4 --samples 64 --variables 8 \
  --output-dir runs/torch_bam

# Test the implementation.
python -m pytest -q tests
```

The smoke test writes `config.json`, `model.pt`, `history.json`, and
`validation.json` into `runs/torch_smoke/`. Output directories are never
overwritten: use `--output-dir runs/torch_smoke_2` to run it again.
Normal training writes the same files, with checkpoint reload/decoding checks
reserved for smoke mode. `model.pt` contains model weights; optimizer state and
resume-training support are not included.

For CPDAG-only training with the original-style Chebyshev generator:

```bash
python train_joint_torch.py --task cpdag --mechanisms chebyshev \
  --epochs 2 --steps-per-epoch 100 --output-dir runs/torch_cpdag
```

Training data are generated on demand; no dataset download is required.
See [PYTORCH_QUICKSTART.md](PYTORCH_QUICKSTART.md) for variable-size batches,
custom training loops, loss masks, and the prediction API. Run either CLI with
`--help` for all arguments.

## Predict

Supply a NumPy array `observations.npy` of shape `[samples, variables]`:

```bash
python predict_joint_torch.py --model-dir runs/torch_bam \
  --input observations.npy --output predictions.npz --target both
```

Use `--target cpdag` to stop before DAG refinement. A checkpoint from the smoke
test can exercise this command but is not a useful trained causal model.

## Files

| File | Purpose |
|---|---|
| `requirements-torch.txt` | PyTorch, NumPy, and pytest dependencies |
| `install_torch.py` | CPU/CUDA installation, index workaround, and runtime checks |
| `run_bam.py`, `MULTI_GPU.md` | Container-compatible launcher and multi-GPU setup |
| `joint_model_torch.py` | Shared directional stream and gated BAM residual |
| `generator_torch.py` | PyTorch dataset returning complete synthetic batches |
| `joint_data_BAM.py`, `chebyshev_numpy.py` | NumPy data generation and target construction |
| `joint_training_torch.py` | Masked multitask losses and training step |
| `train_joint_torch.py` | Training CLI and end-to-end smoke test |
| `predict_joint_torch.py` | Checkpoint loader and prediction CLI |
| `graph_utils_BAM.py`, `joint_decoding.py` | Skeleton, CPDAG, and consistent DAG decoding |
| `tests/` | Model, generator, gradient, graph, and checkpoint checks |
| `PYTORCH_QUICKSTART.md`, `JOINT_MODEL.md` | Usage and architecture details |

## Interpretation and validation

All new adjacency matrices use `A[source, target]`, the transpose of the original
generator convention. Reciprocal entries in a CPDAG mean an undirected edge.
DAG refinement preserves the estimated CPDAG's equivalence class.

DAG supervision is enabled for the nonlinear additive-noise and linear
non-Gaussian generating regimes. It is masked for linear Gaussian and legacy
Chebyshev data by default. Nonlinearity alone does not guarantee identifiability;
the generating assumptions matter. Read [JOINT_MODEL.md](JOINT_MODEL.md) before
interpreting directed estimates as causal conclusions.

The PyTorch-only suite passed 67 tests, with three opt-in distributed cases
skipped. See MULTI_GPU.md for the process-group tests and environment
limitations.
The five-update smoke test reduced its fixed-batch loss from approximately
2.19287 to 2.18260 and reproduced checkpoint predictions exactly on the tested
CPU environment. These checks establish execution, not causal recovery accuracy.
The dense collider head uses cubic memory in variable count; start with small
graphs. There are no pretrained weights or recovery benchmarks in this package.

This is an architectural port, not a TensorFlow checkpoint converter. See the
quickstart for parameterization changes and the first-order-only matrix-log
backward implementation.

## Upstream and license

Based on upstream commit `d8319340dc91a13d46eebf8f1e6e447a182964e5` from
[PhilippFro/BAM](https://github.com/PhilippFro/BAM). The `main` branch contains
BAM2; `BAM_old` preserves the original implementation.
The archive omits Git history, virtual environments, and generated training outputs.

Original work: Philipp Froehlich and Heinz Koeppl,
[Graph Structure Inference with BAM: Neural Dependency Processing via Bilinear Attention](https://neurips.cc/virtual/2024/poster/96766), NeurIPS 2024.
See the [original README](https://github.com/PhilippFro/BAM/blob/BAM_old/README.md)
for citation details and [LICENSE](LICENSE) for the MIT license.
