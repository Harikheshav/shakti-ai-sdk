"""The ahead-of-time compilers' own logic that needs no framework: tools/tvm_aot.py's arena plan."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from tvm_aot import Tensor, plan, up  # noqa: E402


def tensors(spec):
    """spec: (bytes, first step, last step) per intermediate tensor."""
    ts = []
    for i, (n, first, last) in enumerate(spec):
        t = Tensor(f"t{i}", [n], "int8")
        t.kind, t.first, t.last = "arena", first, last
        ts.append(t)
    return ts


def overlaps(a, b):
    return a.first <= b.last and b.first <= a.last


def test_live_tensors_never_share_memory():
    rng = np.random.default_rng(0)
    for _ in range(200):
        spec = []
        for _ in range(int(rng.integers(1, 12))):
            first = int(rng.integers(0, 10))
            spec.append((int(rng.integers(1, 5000)), first, first + int(rng.integers(0, 5))))
        ts = tensors(spec)
        size = plan(ts)
        for t in ts:
            assert t.offset % 16 == 0 and t.offset + t.nbytes <= size
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                if overlaps(a, b):
                    assert a.offset + up(a.nbytes) <= b.offset or b.offset + up(b.nbytes) <= a.offset


def test_a_chain_reuses_memory():
    # a -> b -> c -> d: each tensor lives two steps, so two buffers suffice
    ts = tensors([(1000, 0, 1), (1000, 1, 2), (1000, 2, 3), (1000, 3, 4)])
    assert plan(ts) == 2 * up(1000)
