"""PyTorch joint BAM model: skeleton/collider heads plus optional DAG heads.

Inputs are [batch, samples, variables]. Adjacencies use [source, target] and
colliders use [parent, center, parent]. No TensorFlow imports are needed.

This is an architectural port, not a conversion of TensorFlow checkpoints.
The original BAM operations are retained: channel-group observation attention,
covariance, positive channel mixing, K^T Q K attention, positive-definite
normalization, congruence residuals, and Hadamard-polynomial activations.
Positive softplus parameterizations replace optimizer-projected nonnegativity;
activation coefficients use a softmax instead of a projected simplex.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class _SymmetricLog(torch.autograd.Function):
    """Spectral log with a basis-independent first derivative.

    Differentiating eigenvectors individually is unstable for repeated
    eigenvalues, even though the SPD matrix logarithm is smooth there. The
    Frechet derivative instead uses its Loewner divided-difference matrix.
    Eigenvalues below ``floor`` are clipped; the derivative is zero within the
    clipped region. The nonsmooth boundary at exactly floor uses zero slope.
    Higher derivatives are deliberately unsupported.
    """

    @staticmethod
    def forward(ctx, matrix, floor):
        matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
        values, vectors = torch.linalg.eigh(matrix)
        logs = values.clamp_min(floor).log()
        ctx.save_for_backward(values, vectors, logs)
        ctx.floor = floor
        return (vectors * logs.unsqueeze(-2)) @ vectors.transpose(-1, -2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, vectors, logs = ctx.saved_tensors
        a, b = values.unsqueeze(-1), values.unsqueeze(-2)
        delta = a - b
        scale = torch.maximum(a.abs(), b.abs()).clamp_min(ctx.floor)
        close = delta.abs() <= math.sqrt(torch.finfo(values.dtype).eps) * scale
        above = (a > ctx.floor) & (b > ctx.floor)
        below = (a <= ctx.floor) & (b <= ctx.floor)
        safe_delta = torch.where(delta != 0, delta, torch.ones_like(delta))
        divided = (logs.unsqueeze(-1) - logs.unsqueeze(-2)) / safe_delta
        # For nearby positive eigenvalues the midpoint derivative has O(delta²)
        # relative error and avoids subtraction cancellation. At repetition it
        # is the exact limit 1/lambda, independent of the eigenvector basis.
        midpoint_slope = 2.0 / (a + b).clamp_min(2 * ctx.floor)
        divided = torch.where(close & above, midpoint_slope, divided)
        divided = torch.where(below, torch.zeros_like(divided), divided)
        grad_output = (grad_output + grad_output.transpose(-1, -2)) * 0.5
        eigen_grad = vectors.transpose(-1, -2) @ grad_output @ vectors
        result = vectors @ (divided * eigen_grad) @ vectors.transpose(-1, -2)
        return (result + result.transpose(-1, -2)) * 0.5, None


def symmetric_matrix_log(matrix, eigenvalue_floor=1e-4):
    """Symmetrized SPD log, evaluated in float64 with stable first gradients.

    The return dtype matches ``matrix``. This function intentionally does not
    add coordinate-dependent diagonal jitter, preserving permutation symmetry.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in square matrix dimensions")
    if not matrix.is_floating_point() or eigenvalue_floor <= 0:
        raise ValueError("matrix must be floating point and eigenvalue_floor positive")
    return _SymmetricLog.apply(matrix.to(torch.float64), float(eigenvalue_floor)).to(matrix.dtype)


def _inverse_softplus(value):
    return torch.log(torch.expm1(torch.as_tensor(value)))


class _ObservationEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Linear(1, channels, bias=False)
        self.second = nn.Linear(channels, channels, bias=False)
        self.scale = nn.Parameter(torch.empty(channels))
        nn.init.xavier_normal_(self.first.weight)
        nn.init.kaiming_normal_(self.second.weight, nonlinearity="relu")
        nn.init.normal_(self.scale, std=1 / math.sqrt(channels))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return x + self.second(F.relu(self.first(x))) * self.scale


class _ObservationAttentionHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.scale = nn.Parameter(torch.ones(channels))
        self.channels = channels
        for layer in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        normalized = self.norm(x)
        logits = self.query(normalized) @ self.key(normalized).transpose(-1, -2)
        weights = torch.softmax(logits / math.sqrt(self.channels), dim=-1)
        return x + (weights @ self.value(normalized)) * self.scale


class _ObservationBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        width = channels // heads
        self.feature_heads = nn.ModuleList([_ObservationAttentionHead(width) for _ in range(heads)])
        self.sample_heads = nn.ModuleList([_ObservationAttentionHead(width) for _ in range(heads)])
        self.ffn_first = nn.Linear(channels, channels, bias=False)
        self.ffn_second = nn.Linear(channels, channels, bias=False)
        self.ffn_scale = nn.Parameter(torch.ones(channels))
        nn.init.kaiming_normal_(self.ffn_first.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.ffn_second.weight, nonlinearity="relu")

    def forward(self, x):
        pieces = x.chunk(len(self.feature_heads), dim=-1)
        x = torch.cat([head(piece) for head, piece in zip(self.feature_heads, pieces)], dim=-1)
        # Feature attention uses d as the token axis; sample attention uses M.
        x = x.transpose(1, 2)
        pieces = x.chunk(len(self.sample_heads), dim=-1)
        x = torch.cat([head(piece) for head, piece in zip(self.sample_heads, pieces)], dim=-1)
        x = x.transpose(1, 2)
        centered = x - x.mean(dim=1, keepdim=True)
        # Clamp variance before sqrt, avoiding sqrt'(0) on constant inputs.
        normalized = centered / centered.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
        return x + self.ffn_second(F.relu(self.ffn_first(normalized))) * self.ffn_scale


def _correlation(matrix):
    diagonal = matrix.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)
    inverse = diagonal.rsqrt()
    return matrix * inverse.unsqueeze(-1) * inverse.unsqueeze(-2)


class _BAMHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        keys = torch.empty(channels, channels).uniform_(1e-3 / channels, 2 / channels)
        queries = torch.empty(channels, channels).uniform_(1e-3 / channels, 2 / channels)
        self.raw_keys = nn.Parameter(_inverse_softplus(keys))
        self.raw_queries = nn.Parameter(_inverse_softplus(queries))
        self.raw_scale = nn.Parameter(_inverse_softplus(torch.ones(channels)))

    def forward(self, matrix):
        correlation = _correlation(matrix)
        # Each output channel is a positive combination of PSD input channels.
        keys = torch.einsum("ncij,co->noij", correlation, F.softplus(self.raw_keys))
        queries = torch.einsum("ncij,co->noij", correlation, F.softplus(self.raw_queries))
        pre_attention = keys.transpose(-1, -2) @ queries @ keys
        pre_attention = (pre_attention + pre_attention.transpose(-1, -2)) * 0.5
        # Entrywise exp preserves PSD; scalar stabilization and congruence
        # normalization preserve it too. This is BAM's PD softmax, not rowsoftmax.
        weights = torch.exp(pre_attention - pre_attention.amax(dim=(-2, -1), keepdim=True))
        inverse = weights.sum(dim=-1).clamp_min(1e-8).rsqrt()
        attention = weights * inverse.unsqueeze(-1) * inverse.unsqueeze(-2)
        residual = attention @ correlation @ attention.transpose(-1, -2)
        return matrix + F.softplus(self.raw_scale)[None, :, None, None] * residual


class _BAMBlock(nn.Module):
    def __init__(self, channels, heads, n_exp):
        super().__init__()
        self.heads = nn.ModuleList([_BAMHead(channels // heads) for _ in range(heads)])
        self.activation_logits = nn.Parameter(torch.zeros(n_exp))
        self.raw_activation_scale = nn.Parameter(_inverse_softplus(torch.ones(channels)))
        self.raw_self_scale = nn.Parameter(torch.tensor(math.log(.19 / .11)))

    def forward(self, matrix):
        pieces = matrix.chunk(len(self.heads), dim=1)
        matrix = torch.cat([head(piece) for head, piece in zip(self.heads, pieces)], dim=1)
        correlation = _correlation(matrix)
        coefficients = self.activation_logits.softmax(dim=0)
        power = correlation
        activation = coefficients[0] * power
        for coefficient in coefficients[1:]:
            power = power * correlation
            activation = activation + coefficient * power
        self_scale = .8 + .3 * self.raw_self_scale.sigmoid()
        result = self_scale * matrix + F.softplus(self.raw_activation_scale)[None, :, None, None] * activation
        return (result + result.transpose(-1, -2)) * .5


class JointBAMTorch(nn.Module):
    """One shared observation/BAM model with CPDAG and optional DAG heads.

    The dense collider head uses O(batch*d**3*collider_channels) activations.
    Begin with modest graphs. Outputs are probabilities/ordering scores, not
    identifiability guarantees; use the graph decoders for structural validity.
    """

    def __init__(self, n_channels=20, data_layers=2, cov_layers=2,
                 hidden_channels=32, collider_channels=16, n_heads=5, n_exp=3,
                 gate_init=.01, covariance_ridge=1e-4, include_dag_heads=True):
        super().__init__()
        integers = (n_channels, data_layers, cov_layers, hidden_channels, collider_channels, n_heads, n_exp)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("Channel, head, layer, and polynomial counts must be integers")
        if min(n_channels, hidden_channels, collider_channels, n_heads, n_exp) < 1:
            raise ValueError("Channels, heads, and n_exp must be positive")
        if n_channels % n_heads or min(data_layers, cov_layers) < 0:
            raise ValueError("n_channels must be divisible by n_heads and layer counts nonnegative")
        if not math.isfinite(covariance_ridge) or covariance_ridge <= 0 or not math.isfinite(gate_init):
            raise ValueError("covariance_ridge must be positive and gate_init finite")
        self.config = dict(n_channels=n_channels, data_layers=data_layers, cov_layers=cov_layers,
                           hidden_channels=hidden_channels, collider_channels=collider_channels,
                           n_heads=n_heads, n_exp=n_exp, gate_init=float(gate_init),
                           covariance_ridge=float(covariance_ridge), include_dag_heads=bool(include_dag_heads))
        self.covariance_ridge = covariance_ridge
        self.include_dag_heads = bool(include_dag_heads)
        self.embedding = _ObservationEmbedding(n_channels)
        self.observation_blocks = nn.ModuleList([_ObservationBlock(n_channels, n_heads) for _ in range(data_layers)])
        self.bam_blocks = nn.ModuleList([_BAMBlock(n_channels, n_heads, n_exp) for _ in range(cov_layers)])
        self.observation_norm = nn.LayerNorm(n_channels, eps=1e-3)
        self.source_features = nn.Linear(n_channels, hidden_channels)
        self.target_features = nn.Linear(n_channels, hidden_channels)
        self.node_features = nn.Linear(n_channels, hidden_channels)
        self.directional_projection = nn.Linear(3 * hidden_channels, hidden_channels)
        self.directional_norm = nn.LayerNorm(hidden_channels, eps=1e-3)
        self.bam_projection = nn.Linear(n_channels, hidden_channels, bias=False)
        self.bam_norm = nn.LayerNorm(hidden_channels, eps=1e-3)
        self.bam_gate = nn.Parameter(torch.full((hidden_channels,), float(gate_init)))
        self.pair_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.skeleton_readout = nn.Linear(hidden_channels, 1)
        self.collider_incoming = nn.Linear(hidden_channels, collider_channels, bias=False)
        self.collider_endpoints = nn.Linear(hidden_channels, collider_channels, bias=False)
        self.collider_center = nn.Linear(2 * hidden_channels, collider_channels)
        self.collider_readout = nn.Linear(collider_channels, 1)
        if self.include_dag_heads:
            self.direction_readout = nn.Linear(hidden_channels, 1)
            self.order_hidden = nn.Linear(2 * hidden_channels, hidden_channels)
            self.order_readout = nn.Linear(hidden_channels, 1)

    def get_config(self):
        return dict(self.config)

    def _bam_features(self, h):
        centered = h - h.mean(dim=1, keepdim=True)
        covariance = torch.einsum("nmic,nmjc->ncij", centered, centered) / h.shape[1]
        covariance = covariance + self.covariance_ridge * torch.eye(h.shape[2], device=h.device, dtype=h.dtype)
        for block in self.bam_blocks:
            covariance = block(covariance)
        return symmetric_matrix_log(covariance, self.covariance_ridge).permute(0, 2, 3, 1)

    def _directional_features(self, h):
        normalized = self.observation_norm(h)
        source = torch.tanh(self.source_features(normalized))
        target = torch.tanh(self.target_features(normalized))
        cross = torch.einsum("nmic,nmjc->nijc", source, target) / h.shape[1]
        nodes = torch.tanh(self.node_features(normalized)).mean(dim=1)
        first = nodes[:, :, None, :].expand_as(cross)
        second = nodes[:, None, :, :].expand_as(cross)
        return self.directional_norm(F.gelu(self.directional_projection(torch.cat((first, second, cross), dim=-1))))

    @staticmethod
    def _node_context(pairs, off_diagonal):
        masked = pairs * off_diagonal[None, :, :, None]
        denominator = off_diagonal.sum(-1).clamp_min(1)[None, :, None]
        return torch.cat((masked.sum(2) / denominator, masked.sum(1) / denominator), dim=-1)

    def forward(self, observations):
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if observations.ndim != 3 or min(observations.shape) < 1:
            raise ValueError("observations must have shape [batch,samples,variables] with nonempty dimensions")
        if not observations.is_floating_point() or not torch.isfinite(observations).all():
            raise ValueError("observations must be finite floating-point values")
        h = self.embedding(observations)
        for block in self.observation_blocks:
            h = block(h)
        directional = self._directional_features(h)
        bam = self.bam_norm(self.bam_projection(self._bam_features(h)))
        pairs = F.gelu(self.pair_hidden(directional + self.bam_gate * bam))
        d = observations.shape[2]
        off_diagonal = 1 - torch.eye(d, device=pairs.device, dtype=pairs.dtype)
        node_context = self._node_context(pairs, off_diagonal)
        logits = self.skeleton_readout(pairs).squeeze(-1)
        skeleton = torch.sigmoid((logits + logits.transpose(1, 2)) * .5) * off_diagonal
        incoming = self.collider_incoming(pairs)
        endpoints = self.collider_endpoints((pairs + pairs.transpose(1, 2)) * .5)
        center = self.collider_center(node_context)
        triples = (incoming[:, :, :, None, :] + incoming.transpose(1, 2)[:, None, :, :, :]
                   + endpoints[:, :, None, :, :] + center[:, None, :, None, :])
        collider_logits = self.collider_readout(torch.tanh(triples)).squeeze(-1)
        collider_logits = (collider_logits + collider_logits.transpose(1, 3)) * .5
        distinct = off_diagonal[:, :, None] * off_diagonal[None, :, :] * off_diagonal[:, None, :]
        result = {"skeleton": skeleton, "colliders": collider_logits.sigmoid() * distinct}
        if self.include_dag_heads:
            logits = self.direction_readout(pairs).squeeze(-1)
            result["direction"] = (logits - logits.transpose(1, 2)).sigmoid() * off_diagonal
            order = self.order_readout(F.gelu(self.order_hidden(node_context))).squeeze(-1)
            result["order"] = order - order.mean(dim=-1, keepdim=True)
        return result


JointBAM = JointBAMTorch
