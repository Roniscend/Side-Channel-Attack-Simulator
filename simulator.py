import numpy as np
from aes_core import hamming_weight
from leakage_models import _HW_SBOX

TRACE_LEN  = 2000
LEAK_START = 400
LEAK_WIDTH = 50
LEAK_SCALE = 0.8


def _build_hw_matrix(plaintexts: np.ndarray, key: bytes) -> np.ndarray:
    key_arr = np.frombuffer(key, dtype=np.uint8)
    xor = plaintexts.astype(np.uint16) ^ key_arr[None, :]
    xor = (xor & 0xFF).astype(np.uint8)
    return _HW_SBOX[xor].astype(np.float32)


def generate_traces(
    key: bytes,
    n_traces: int,
    seed: int = 42,
    noise_std: float = 1.5,
    jitter: int = 0,
    masking: bool = False,
    shuffling: bool = False,
    verbose: bool = True,
) -> tuple:
    rng = np.random.default_rng(seed)
    plaintexts = rng.integers(0, 256, size=(n_traces, 16), dtype=np.uint8)
    traces = rng.standard_normal((n_traces, TRACE_LEN)).astype(np.float32) * noise_std

    key_arr = np.frombuffer(key, dtype=np.uint8)

    if masking:
        masks = rng.integers(0, 256, size=(n_traces, 16), dtype=np.uint8)
        masked_pt = (plaintexts.astype(np.uint16) ^ masks) & 0xFF
        xor1 = (masked_pt.astype(np.uint16) ^ key_arr[None, :]) & 0xFF
        from aes_core import SBOX as _SBOX
        sbox_arr = np.array(_SBOX, dtype=np.uint8)
        sbox_out = sbox_arr[xor1.astype(np.uint8)]
        remasked = (sbox_out.astype(np.uint16) ^ masks) & 0xFF
        hw_leak = _HW_SBOX[remasked.astype(np.uint8)].astype(np.float32)
    else:
        hw_leak = _build_hw_matrix(plaintexts, key)

    for byte_idx in range(16):
        base_pos = LEAK_START + byte_idx * LEAK_WIDTH
        offsets = rng.integers(-jitter, jitter + 1, size=n_traces) if jitter > 0 else np.zeros(n_traces, dtype=int)

        if shuffling:
            perm_offsets = rng.integers(0, 16, size=n_traces) * LEAK_WIDTH
            offsets = offsets + perm_offsets

        positions = np.clip(base_pos + offsets, 0, TRACE_LEN - LEAK_WIDTH)

        for i in range(n_traces):
            p = positions[i]
            traces[i, p:p + LEAK_WIDTH] += hw_leak[i, byte_idx] * LEAK_SCALE

    return plaintexts, traces


def compute_snr(plaintexts: np.ndarray, traces: np.ndarray, key: bytes) -> np.ndarray:
    hw_matrix = _build_hw_matrix(plaintexts, key)
    hw = hw_matrix[:, 0].astype(int)

    noise_var = np.zeros(traces.shape[1], dtype=np.float64)
    unique_hw = np.unique(hw)
    class_means = []

    for h in unique_hw:
        mask = hw == h
        class_traces = traces[mask]
        class_means.append(class_traces.mean(axis=0))
        noise_var += mask.sum() * class_traces.var(axis=0)

    class_means = np.array(class_means)
    signal_var = class_means.var(axis=0)
    noise_var /= len(hw)

    return (signal_var / (noise_var + 1e-12)).astype(np.float32)
