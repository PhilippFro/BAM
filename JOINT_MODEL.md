# Joint BAM: skeleton → CPDAG → DAG

This experimental extension uses **one jointly trained network**, followed by
three inference stages. It preserves BAM's CPDAG target and adds an optional,
mechanism-dependent DAG refinement. No pretrained weights are supplied, and
test coverage establishes implementation properties rather than scientific
recovery accuracy.

**PyTorch is now available and tested end to end.** Start with
[PYTORCH_QUICKSTART.md](PYTORCH_QUICKSTART.md) for its simple generator, training
smoke test, and prediction commands. The architecture and decoding descriptions
below apply to both implementations; TensorFlow-specific commands are retained
as a reference.

## Architecture

`joint_model_BAM.JointBAM` reuses the original observation-attention and SPD BAM
layers. It splits the shared observation encoding into:

- A directional path with nonlinear node summaries and ordered source/target
  cross-moments, pooled over samples.
- A covariance → BAM → LogEig path, projected and normalized into pair features.

The fused representation is `directional + gamma * bam`, where `gamma` is a
learnable channel vector initialized to `0.01`. The gate comes **after** LogEig
and normalization. Nonlinear heads follow fusion. All variable pairs share the
same parameters; the network has no variable-position embeddings.

The heads predict:

| Output | Shape | Meaning |
|---|---|---|
| `skeleton` | `[batch,d,d]` | Symmetric probability of a direct skeleton edge |
| `colliders` | `[batch,d,d,d]` | Probability of `i → j ← k`, axes `[i,j,k]`, symmetric in endpoints |
| `direction` | `[batch,d,d]` | Conditional orientation preference given an edge; off-diagonal reciprocal probabilities sum to one |
| `order` | `[batch,d]` | Auxiliary node scores, with parents encouraged to precede children |

The DAG heads can be omitted with `include_dag_heads=False`. Collider scores are
not masked using predicted skeleton edges during training. Their loss uses the
true unshielded-triple mask, avoiding a hard predicted-candidate bottleneck.

## Three inference stages

`predict_joint_BAM.predict` calls the network once and then decodes its scores:

1. **Skeleton:** threshold symmetric edge probabilities and remove self-edges.
2. **CPDAG:** resolve potentially incompatible collider predictions by searching
   over DAG witnesses on that skeleton. A bounded beam scores unshielded triples
   by collider/noncollider likelihood. Convert the selected witness to its exact
   CPDAG using compelled-edge labeling. This yields a valid CPDAG even when local
   collider predictions conflict. The internal witness is not an additional
   causal prediction, and this stage never uses mechanism-dependent direction
   or ordering scores.
3. **DAG:** use direction preferences to choose a consistent DAG extension of
   that CPDAG. Score-guided sink elimination preserves its skeleton, compelled
   arrows, and unshielded colliders. Verify `dag_to_cpdag(dag) == cpdag` before
   returning. Order scores can break directional-score ties.

CPDAG conversion and the graph-validity guarantees are exact. The finite beam
and score-guided extension are **heuristics**, not globally optimal score
solvers. Beam width controls the quality/runtime tradeoff. Ties can produce
different equally scored graph estimates after variable relabeling even though
the neural scores themselves are permutation-equivariant.

The final stage cannot correct errors already committed in the CPDAG. For
example, a wrongly compelled arrow stays fixed. Keep all three outputs for
evaluation rather than reporting only the final DAG. `graph_utils_BAM.decode_dag`
also exposes an independent DAG decoder for explicit ablation experiments;
the default inference pipeline uses CPDAG-consistent refinement.

## Identifiability and supervision

A CPDAG represents a **conditional-independence equivalence class**; an
undirected edge is not simply an edge with low neural confidence. This target
is meaningful beyond Gaussian data. Nonlinearity or non-Gaussian marginals alone
do not guarantee a uniquely identifiable DAG.

Additional mechanism assumptions can identify directions inside the CPDAG's
equivalence class. Examples are linear models with independent non-Gaussian
noise and suitable nonlinear additive-noise models, even with Gaussian noise.
The DAG output is a point estimate under the training regime's assumptions;
its confidence is not an identifiability certificate.

`joint_data_BAM` supplies illustrative causal-sufficient, independent-noise
simulation regimes:

| Regime | CPDAG supervision | DAG supervision by default |
|---|---|---|
| `linear_gaussian` with varying noise variances | Yes | No |
| `linear_nongaussian` with independent Laplace noise | Yes | Yes |
| `nonlinear_anm` with smooth nonlinear parent functions and independent additive Gaussian noise | Yes | Yes |
| `chebyshev` using the existing generator | Yes | No; its unrestricted mechanisms are not assumed identifiable |

`dag_mask` is explicit per-dataset metadata. The loss masks **both** direction
and ordering objectives when it is zero. Synthetic generating DAG labels are
retained for controlled evaluation, but their mere availability does not enable
DAG supervision. Callers may deliberately override `dag_supervision` when they
intend different assumptions; the model does not infer that permission from
whether samples look nonlinear.

Training combines unordered-pair skeleton BCE, unshielded-collider BCE,
conditional direction BCE on true edges, and parent-before-child hinge loss.
Only actual DAG edges impose ordering constraints; unrelated variables receive
no arbitrary total-order target. Losses are averaged per eligible graph so
graphs with more variables/triples do not automatically dominate.

## TensorFlow reference: install and train

Use a separate Python 3.10–3.12 environment. The extension retains Keras 2
compatibility for the original layers:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-joint.txt
export TF_USE_LEGACY_KERAS=1
```

Set the environment variable before any TensorFlow/Keras imports in notebooks.
The new entry points set it by default when run in a fresh process.

Train all heads using a mix of identifiable and ambiguous regimes:

```bash
python train_joint_BAM.py --task both \
  --mechanisms linear_gaussian nonlinear_anm linear_nongaussian \
  --variables 16 --samples 128 --batch-size 4 \
  --epochs 10 --steps-per-epoch 128 --output-dir runs/joint_bam
```

For CPDAG-only training:

```bash
python train_joint_BAM.py --task cpdag \
  --mechanisms linear_gaussian nonlinear_anm chebyshev \
  --output-dir runs/cpdag_bam
```

These are starting configurations, not tuned recommendations. A short
`--epochs 1 --steps-per-epoch 2 --variables 4 --samples 12 --batch-size 2`
run checks the pipeline; it does not train a useful causal estimator.
Training saves architecture configuration, weights, and loss history. Existing
model directories are not overwritten. `--task dag` keeps the skeleton/collider
objectives as well, because the inference path still passes through a CPDAG.

The plain `JointDataGenerator` API supports varying sample and variable counts;
pass inclusive pairs such as `num_variables=(4, 20)`. The CLI currently trains
at one size for a reproducible baseline. Robust size generalization requires
varying training sizes and separate evaluation.

## TensorFlow reference: predict

Save one observed data matrix as `.npy` with shape `[samples,variables]`, or as
`.npz` under key `X`. Columns must have the same interpretation across rows.

```bash
python predict_joint_BAM.py --model-dir runs/joint_bam \
  --input observations.npy --output predictions.npz --target both
```

Use `--target skeleton` or `--target cpdag` to stop early. `--target dag` still
uses the CPDAG stage. DAG refinement requires a model trained with the DAG heads.
Inputs are standardized by column as in training; `--no-standardize` is an
explicit override. Prediction files contain decoded graphs, raw neural scores,
and decoder metadata. Existing output files are not overwritten.

Python usage:

```python
from predict_joint_BAM import load_model, predict

model = load_model("runs/joint_bam")
graphs, scores = predict(model, observations, target="both")
cpdag = graphs["cpdag"]
dag = graphs["dag"]
```

## Matrix conventions

All new graph APIs use **`A[source,target] == 1`**. This is the transpose of the
legacy generator's convention. The Chebyshev adapter performs that conversion
explicitly and keeps the original graph unmodified.

In a CPDAG, `C[i,j]=C[j,i]=1` means an undirected edge; `C[i,j]=1,C[j,i]=0`
means `i→j`. Consequently, a CPDAG matrix must not be passed to a directed-cycle
test that interprets an undirected edge as a two-cycle. Collider tensors use
`colliders[parent,center,parent]`; the first and last axes are interchangeable.

## Validation and limitations

```bash
python -m pytest -q tests
```

Tests include exact CPDAG comparison against enumerated four-node equivalence
classes, five-node cases, incompatible collider scores, CPDAG-consistent DAG
refinement, supervision masks, permutation properties, gradients, optimizer
updates, dynamic shapes, and weight-save/reload behavior.

For PyTorch, **68 tests pass**, including actual neural forward/backward,
training updates, and checkpoint tests; two TensorFlow-only modules are skipped.
See the PyTorch guide for the verified smoke-test results.

Earlier TensorFlow validation in the implementation workspace: **30 pure NumPy graph/data/inference
tests passed**, and all Python files passed syntax compilation. The neural
tests are provided but could not be executed: TensorFlow 2.16.2 aborted while
initializing kernels because this workspace does not expose the `/proc/self`
process metadata it expects. Consequently, numerical forward/backward passes,
optimizer updates, and checkpoint round trips remain unverified. No model was
trained. Run the complete suite in a normal TensorFlow environment before using
the training results. To reproduce the verified subset:

```bash
python -m pytest -q tests/test_graph_utils.py tests/test_joint_data.py tests/test_joint_inference.py
```

The collider head currently materializes `O(batch * d^3 * collider_channels)`
features. Start with modest graphs; batching or sampling candidate triples is
needed for larger scales. The original sample-attention cost also remains.
There are no recovery benchmarks, calibrated thresholds, latent-confounder
support, or causal-identifiability tests in this prototype. Full evaluations
should compare CPDAG recovery on common targets and DAG recovery separately in
identifiable regimes, with held-out mechanisms, graph sizes, and sample sizes.

References: [BAM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e8ccaec7aea6f5740a3a3fdba390e466-Abstract-Conference.html),
[additive-noise identifiability](https://jmlr.org/papers/v15/peters14a.html),
[LiNGAM](https://jmlr.org/papers/v7/shimizu06a.html), and
[efficient equivalence-class identification](https://jmlr.org/papers/v2/chickering02a.html).

Single-server multi-GPU training, globally normalized masked losses, and the
container launcher are described in [MULTI_GPU.md](MULTI_GPU.md). Real process-group
and CUDA tests require the target server; they could not be run in this environment.
