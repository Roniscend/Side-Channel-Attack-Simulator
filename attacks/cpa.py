import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from leakage_models import MODELS, hw_model, _HW_SBOX


def _pearson_matrix(H: np.ndarray, T: np.ndarray) -> np.ndarray:
    H_c = H - H.mean(axis=0, keepdims=True)
    T_c = T - T.mean(axis=0, keepdims=True)
    num   = H_c.T @ T_c
    H_std = np.sqrt((H_c ** 2).sum(axis=0))
    T_std = np.sqrt((T_c ** 2).sum(axis=0))
    denom = np.outer(H_std, T_std) + 1e-12
    return np.abs(num / denom)


def _build_H(pt_col: np.ndarray, model_fn) -> np.ndarray:
    H = np.empty((len(pt_col), 256), dtype=np.float32)
    for k in range(256):
        H[:, k] = model_fn(pt_col, k)
    return H


def attack_byte(byte_idx: int, plaintexts: np.ndarray, traces: np.ndarray, model: str = "hw") -> dict:
    model_fn = MODELS.get(model, hw_model)
    pt_col   = plaintexts[:, byte_idx]
    H        = _build_H(pt_col, model_fn)
    abs_corr = _pearson_matrix(H, traces.astype(np.float32))

    peak_per_guess = abs_corr.max(axis=1)
    sorted_peaks   = np.sort(peak_per_guess)[::-1]
    best_guess     = int(peak_per_guess.argmax())
    confidence     = float(sorted_peaks[0] - sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0

    return {
        "best_guess":     best_guess,
        "max_corr":       float(peak_per_guess[best_guess]),
        "corr_matrix":    abs_corr,
        "peak_per_guess": peak_per_guess,
        "confidence":     confidence,
    }


def recover_full_key(plaintexts: np.ndarray, traces: np.ndarray, model: str = "hw", verbose: bool = True) -> tuple:
    recovered, results = [], []
    for b in range(16):
        res = attack_byte(b, plaintexts, traces, model=model)
        recovered.append(res["best_guess"])
        results.append(res)
        if verbose:
            print(f"  Byte {b:2d}: 0x{res['best_guess']:02x}  "
                  f"corr={res['max_corr']:.4f}  margin={res['confidence']:.4f}")
    return bytes(recovered), results


def incremental_cpa(byte_idx: int, plaintexts: np.ndarray, traces: np.ndarray, steps: int = 20, model: str = "hw") -> np.ndarray:
    n = len(traces)
    checkpoints = np.linspace(10, n, steps, dtype=int)
    model_fn = MODELS.get(model, hw_model)
    pt_col   = plaintexts[:, byte_idx]

    results = []
    for cnt in checkpoints:
        H = _build_H(pt_col[:cnt], model_fn)
        abs_corr = _pearson_matrix(H, traces[:cnt].astype(np.float32))
        results.append(abs_corr.max(axis=1))

    return np.array(results), checkpoints
