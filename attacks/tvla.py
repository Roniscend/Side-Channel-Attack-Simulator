import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TVLA_THRESHOLD = 4.5


def run_tvla(
    key: bytes,
    n_traces: int = 1000,
    seed: int = 0,
    noise_std: float = 1.5,
    jitter: int = 0,
    masking: bool = False,
    shuffling: bool = False,
) -> dict:
    from simulator import generate_traces

    rng = np.random.default_rng(seed)
    fixed_pt = bytes(rng.integers(0, 256, 16, dtype=np.uint8))

    fixed_traces = _generate_fixed(key, fixed_pt, n_traces, seed + 1, noise_std, jitter, masking, shuffling)

    _, random_traces = generate_traces(
        key, n_traces, seed=seed + 2,
        noise_std=noise_std, jitter=jitter,
        masking=masking, shuffling=shuffling, verbose=False,
    )

    t_stat = _welch_t(fixed_traces, random_traces)
    leaky_mask = np.abs(t_stat) > TVLA_THRESHOLD

    return {
        "t_stat":      t_stat,
        "leaky":       bool(leaky_mask.any()),
        "leaky_mask":  leaky_mask,
        "fixed_mean":  fixed_traces.mean(axis=0),
        "random_mean": random_traces.mean(axis=0),
        "threshold":   TVLA_THRESHOLD,
    }


def _generate_fixed(key, fixed_pt, n_traces, seed, noise_std, jitter, masking, shuffling):
    from leakage_models import _HW_SBOX
    from simulator import TRACE_LEN, LEAK_START, LEAK_WIDTH, LEAK_SCALE

    rng = np.random.default_rng(seed)
    pt_arr = np.frombuffer(fixed_pt, dtype=np.uint8)
    plaintexts = np.tile(pt_arr, (n_traces, 1))
    traces = rng.standard_normal((n_traces, TRACE_LEN)).astype(np.float32) * noise_std

    key_arr = np.frombuffer(key, dtype=np.uint8)
    xor = (plaintexts.astype(np.uint16) ^ key_arr[None, :]) & 0xFF
    hw_leak = _HW_SBOX[xor.astype(np.uint8)].astype(np.float32)

    for byte_idx in range(16):
        base_pos = LEAK_START + byte_idx * LEAK_WIDTH
        offsets = rng.integers(-jitter, jitter + 1, size=n_traces) if jitter > 0 else np.zeros(n_traces, int)
        positions = np.clip(base_pos + offsets, 0, TRACE_LEN - LEAK_WIDTH)
        for i in range(n_traces):
            p = positions[i]
            traces[i, p:p + LEAK_WIDTH] += hw_leak[i, byte_idx] * LEAK_SCALE

    return traces


def _welch_t(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n1, n2 = A.shape[0], B.shape[0]
    m1, m2 = A.mean(axis=0), B.mean(axis=0)
    v1, v2 = A.var(axis=0, ddof=1), B.var(axis=0, ddof=1)
    return (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2 + 1e-12)
