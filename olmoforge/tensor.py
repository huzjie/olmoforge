"""Zero-dependency tensor core with a small autodiff graph.

This is deliberately small: it only implements the ops used by the MoE layer,
the router, the load-balancing loss, and the trainer. It keeps the whole
project runnable in a bare Python 3.9+ interpreter with no third-party
packages.

Representation: a Tensor wraps `data` as nested lists (rows), carries
`requires_grad` and `grad`, and records `_ctx` for backprop. Ops create new
Tensors whose `_backward` closes over their inputs.
"""
import math


def _flatten(x):
    if isinstance(x, Tensor):
        return _flatten(x.data)
    if isinstance(x, (list, tuple)):
        return [e for row in x for e in _flatten(row)]
    return [x]


class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        if isinstance(data, Tensor):
            data = data.data
        if data and not isinstance(data[0], (list, tuple)):
            data = [list(data)]
        self.data = [list(row) for row in data]
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self.name = name

    # -- shape helpers -------------------------------------------------
    @property
    def shape(self):
        if not self.data:
            return (0,)
        return (len(self.data), len(self.data[0]))

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data!r})"

    def item(self):
        flat = _flatten(self.data)
        if len(flat) != 1:
            raise ValueError("item() only for single-element tensors")
        return flat[0]

    def zeros_like(self):
        return Tensor([[0.0] * len(r) for r in self.data])

    def ones_like(self):
        return Tensor([[1.0] * len(r) for r in self.data])

    def backward(self, grad=None):
        if grad is None:
            grad = [[1.0] * len(r) for r in self.data]
        self.grad = grad
        topo = []
        seen = set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            if t._ctx is not None:
                for inp in t._ctx["inputs"]:
                    build(inp)
                topo.append(t)

        build(self)
        for t in reversed(topo):
            t._ctx["backward"](t.grad)

    def _as_ctx(self, inputs, backward):
        self._ctx = {"inputs": inputs, "backward": backward}

    # -- elementwise ops ----------------------------------------------
    def _binop(self, other, op, bop):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = [[op(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)]
        t = Tensor(out, requires_grad=self.requires_grad or other.requires_grad)
        if t.requires_grad:
            def _back(g):
                if self.requires_grad:
                    self.grad = g if self.grad is None else _add_grads(self.grad, g)
                if other.requires_grad:
                    og = bop(g, other.data)
                    other.grad = og if other.grad is None else _add_grads(other.grad, og)
            t._as_ctx([self, other], _back)
        return t

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b, lambda g, _: g)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b, lambda g, _: [[-x for x in r] for r in g])

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b,
                           lambda g, od: [[g[i][j] * od[i][j] for j in range(len(od[i]))]
                                          for i in range(len(od))])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b,
                           lambda g, od: [[g[i][j] / od[i][j] for j in range(len(od[i]))]
                                          for i in range(len(od))])

    def relu(self):
        out = [[max(0.0, x) for x in r] for r in self.data]
        t = Tensor(out, requires_grad=self.requires_grad)
        if t.requires_grad:
            def _back(g):
                self.grad = [[g[i][j] if self.data[i][j] > 0 else 0.0
                              for j in range(len(g[i]))] for i in range(len(g))]
            t._as_ctx([self], _back)
        return t

    def gelu(self):
        out = [[0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
               (v + 0.044715 * v ** 3))) for v in row] for row in self.data]
        t = Tensor(out, requires_grad=self.requires_grad)
        if t.requires_grad:
            def _back(g):
                from math import tanh, sqrt, pi
                c = sqrt(2.0 / pi)
                d = [[0.5 * (1.0 + tanh(c * (v + 0.044715 * v ** 3))
                     + c * v * (1 + 3 * 0.044715 * v ** 2)
                     * (1 - tanh(c * (v + 0.044715 * v ** 3)) ** 2))
                     for v in row] for row in self.data]
                self.grad = [[g[i][j] * d[i][j] for j in range(len(g[i]))]
                             for i in range(len(g))]
            t._as_ctx([self], _back)
        return t

    def sum(self):
        out = sum(_flatten(self.data))
        t = Tensor([[out]], requires_grad=self.requires_grad)
        if t.requires_grad:
            def _back(g):
                self.grad = [[g[0][0]] * len(r) for r in self.data]
            t._as_ctx([self], _back)
        return t

    def mean(self):
        n = len(_flatten(self.data))
        s = self.sum()
        return s * (1.0 / n)

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = [[sum(a * b for a, b in zip(row, col)) for col in _cols(other.data)]
               for row in self.data]
        t = Tensor(out, requires_grad=self.requires_grad or other.requires_grad)
        if t.requires_grad:
            def _back(g):
                if self.requires_grad:
                    gt = _matmul(g, _transpose(other.data))
                    self.grad = gt if self.grad is None else _add_grads(self.grad, gt)
                if other.requires_grad:
                    og = _matmul(_transpose(self.data), g)
                    other.grad = og if other.grad is None else _add_grads(other.grad, og)
            t._as_ctx([self, other], _back)
        return t


def _cols(m):
    return [[row[j] for row in m] for j in range(len(m[0]))]


def _transpose(m):
    return _cols(m)


def _matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in _cols(b)] for row in a]


def _add_grads(g1, g2):
    return [[g1[i][j] + g2[i][j] for j in range(len(g1[i]))] for i in range(len(g1))]


def softmax(x, dim=-1):
    """Numerically stable softmax over the last dim (returns plain nested list)."""
    if x and isinstance(x[0], (list, tuple)):
        return [softmax(row, dim) for row in x]
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [e / s for e in ex]


def layernorm(x, eps=1e-5):
    """LayerNorm over the last dim, supporting 1D and 2D (token sequences)."""
    if x and isinstance(x[0], (list, tuple)):
        return [layernorm(row, eps) for row in x]
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * inv for v in x]


def gelu(x):
    if isinstance(x, (list, tuple)):
        return [gelu(v) for v in x]
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
