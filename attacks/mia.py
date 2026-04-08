import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from leakage_models import MODELS, hw_model


def _mutual_information(h: np.ndarray, t: np.ndarray, bins: int = 20) -> float:
    h_int = h.astype(int)
    h_min, h_max = h_int.min(), h_int.max()
    if h_min == h_max:
        return 0.0

    h_bins = np.arange(h_min, h_max + 2) - 0.5
    t_bins = np.linspace(t.min(), t.max() + 1e-9, bins + 1)

    joint, _, _ = np.histogram2d(h_int, t, bins=[h_bins, t_bins])
    joint = joint / joint.sum()

    p_h = joint.sum(axis=1, keepdims=True)
    p_t = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(joint > 0, joint / (p_h * p_t + 1e-300), 0.0)
        mi = np.sum(joint * np.log2(ratio + 1e-300))

    return float(max(mi, 0.0))


def attack_byte(
    byte_idx: int,
    plaintexts: np.ndarray,
    traces: np.ndarray,
    model: str = "hw",
    bins: int = 20,
    sample_step: int = 5,
) -> dict:
    model_fn = MODELS.get(model, hw_model)
    pt_col   = plaintexts[:, byte_idx]
    T        = traces.astype(np.float32)
    sample_indices = np.arange(0, T.shape[1], sample_step)
    mi_matrix = np.zeros((256, len(sample_indices)), dtype=np.float32)

    for k in range(256):
        h = model_fn(pt_col, k)
        for j, s in enumerate(sample_indices):
            mi_matrix[k, j] = _mutual_information(h, T[:, s], bins=bins)

    peak_per_guess = mi_matrix.max(axis=1)
    sorted_peaks   = np.sort(peak_per_guess)[::-1]
    best_guess     = int(peak_per_guess.argmax())
    confidence     = float(sorted_peaks[0] - sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0

    return {
        "best_guess":     best_guess,
        "max_mi":         float(peak_per_guess[best_guess]),
        "mi_matrix":      mi_matrix,
        "peak_per_guess": peak_per_guess,
        "confidence":     confidence,
        "sample_indices": sample_indices,
    }


def recover_full_key(
    plaintexts: np.ndarray,
    traces: np.ndarray,
    model: str = "hw",
    bins: int = 20,
    sample_step: int = 10,
    verbose: bool = True,
) -> tuple:
    recovered, results = [], []
    for b in range(16):
        res = attack_byte(b, plaintexts, traces, model=model, bins=bins, sample_step=sample_step)
        recovered.append(res["best_guess"])
        results.append(res)
        if verbose:
            print(f"  Byte {b:2d}: 0x{res['best_guess']:02x}  "
                  f"peak_MI={res['max_mi']:.4f}  margin={res['confidence']:.4f}")
    return bytes(recovered), results
