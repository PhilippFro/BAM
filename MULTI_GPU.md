# Single-server multi-GPU training

The training CLI supports one CPU process, one CUDA GPU, or one worker per GPU
with PyTorch DistributedDataParallel (DDP). CUDA workers use NCCL; CPU workers
use Gloo. This guide targets one server with 4 or 8 GPUs. Multi-node launches
are rejected by the training entry point.

## Install the CUDA build in your existing virtual environment

Extract `BAM2.zip`, enter `BAM2`, and activate your `.venv-torch`. If you need
a new environment, create one with `python -m venv .venv-torch` and activate it
with `source .venv-torch/bin/activate`. The old
CPU wheel (`2.14.1+cpu`) cannot use CUDA. For the reported A100 server with driver
610.43.02, install the official CUDA 13.0 build:

```bash
python -E -s run_bam.py install_torch.py --clean-indexes --cuda
```

This replaces only the PyTorch wheel in the active Python environment with
`torch==2.14.1+cu130`, installs its dependencies, and checks CPU and CUDA backward
passes. It prints the visible GPU count. It does not reinstall Conda, change the
system NVIDIA driver, or require a matching system CUDA toolkit installation.
The CUDA version displayed by `nvidia-smi` is not the PyTorch wheel version.
For a different server, check its driver separately before using this build.

The official [wheel index](https://download.pytorch.org/whl/cu130/torch/) lists
Python 3.14 Linux x86-64 wheels for this version. NVIDIA's
[compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
requires driver >=580 for CUDA 13.x; the supplied driver exceeds this.

`run_bam.py` addresses the earlier Python 3.10/Python 3.14 native-library conflict:
it prefers the active environment's `torch/lib`, removes old Torch library paths
and Torch preloads, and clears Python path overrides for the child process.
It retains unrelated library paths, including NVIDIA driver paths, and
`CUDA_VISIBLE_DEVICES`. It also forces torchrun to use the same Python interpreter.
No shell configuration or system packages are changed. Invoke it with `-E -s`
as above so the launcher itself also ignores Python environment overrides.
It can also launch `install_torch.py` in a new empty virtual environment; the
command above installs everything needed before running its runtime checks.

## First check one GPU

```bash
python -E -s run_bam.py train_joint_torch.py \
  --device cuda:0 --smoke-test --output-dir runs/gpu_smoke
```

`--smoke-test` now honors `--device`. It still defaults to CPU if the device is
omitted. It runs five tiny updates, saves/reloads the checkpoint, and validates
the skeleton → CPDAG → DAG decode. It is an execution check, not an accuracy
benchmark. The small workload may have low GPU utilization.

## Check all four GPUs

```bash
python -E -s run_bam.py -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --local-addr=127.0.0.1 \
  train_joint_torch.py --device cuda --smoke-test \
  --output-dir runs/ddp_smoke_4gpu
```

For the other server, use `--nproc-per-node=8` and a new output directory. Under
DDP, use `--device cuda` without an index. Each worker selects its local GPU.
The smoke test also checks that all replicas have matching parameters.
Only worker 0 prints training metrics and writes the checkpoint and JSON files.
Launcher/backend messages may still appear from other workers.

## Train on four GPUs

```bash
python -E -s run_bam.py -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --local-addr=127.0.0.1 \
  train_joint_torch.py --device cuda \
  --epochs 5 --steps-per-epoch 100 \
  --batch-size 8 --samples 128 --variables 12 \
  --output-dir runs/training_4gpu
```

`--batch-size` is **per GPU**: this command trains on 32 synthetic datasets per
optimizer update. On 8 GPUs, the same setting gives 64. To keep the global batch
at 32 when moving to 8 GPUs, use `--batch-size 4`. The learning rate does not
change automatically. Every worker performs the same number of updates, using
a different reproducible generator seed. Epochs advance each worker's data stream.

The defaults are starting settings, not tuned for maximum A100 throughput.
Generation and graph decoding run on CPU; neural forward/backward operations
run on the selected GPU. DDP copies the full model onto each GPU: it increases
batch throughput, not the maximum size of one graph. Dense collider features
still grow cubically with the variable count. Small graphs/batches may run
slower with extra GPUs because synchronization has a cost. Mixed precision is
not enabled; the stable matrix-log calculation retains float64 internally.

To select two GPUs on a four-GPU host:

```bash
CUDA_VISIBLE_DEVICES=1,3 python -E -s run_bam.py -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --local-addr=127.0.0.1 \
  train_joint_torch.py --device cuda --epochs 5 \
  --output-dir runs/training_2gpu
```

## Objectives and checkpoints

Each loss is normalized by the number of eligible graphs across **all** workers.
This matters when some batches contain only linear-Gaussian data or no unshielded
triples. A worker with no eligible direction targets contributes zero direction
gradient without diluting other workers' supervised direction loss. The local
loss gradient is scaled to account for DDP's gradient averaging. Reported losses
are the global values; supervised-dataset counts are summed across workers.

Checkpoints contain the unwrapped model state, so the existing CPU/single-GPU
prediction script loads them without changes. Config records world size,
per-worker and global batch sizes, backend, and data seeds. Optimizer state and
resume training are not implemented. Existing output directories remain protected.

## Validation

```bash
python -E -s run_bam.py -m pytest -q tests

# On a server permitting local sockets, also exercise two-process CPU DDP
# and (when two GPUs are visible) two-GPU NCCL DDP:
BAM_DISTRIBUTED_TESTS=1 python -E -s run_bam.py -m pytest -q tests/test_torch_distributed.py
```

Local validation covers the ordinary CPU pipeline, loss values and averaged
per-worker model gradients against a combined-batch reference (including empty
supervision on one worker), and preservation of driver paths in the launcher.
The true process-group tests are opt-in. This development environment has no
CUDA GPUs and blocks Gloo's communication sockets, so neither actual DDP
training nor GPU execution has been validated here. Run the GPU smoke tests
above on your server before starting a long job. No multi-GPU speedup is claimed.

If installation succeeds but CUDA is unavailable, retain the launcher's printed
PyTorch path, CUDA version, and full traceback. If NCCL communication fails,
rerun the small smoke command with `NCCL_DEBUG=INFO` prefixed and inspect the
error; do not disable transports or change driver settings without evidence.

Implementation references:
[torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html) and
[DistributedDataParallel](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).
