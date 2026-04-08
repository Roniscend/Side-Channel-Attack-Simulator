import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from leakage_models import _HW_SBOX


def _selection_bit(pt_col: np.ndarray, key_guess: int, bit: int = 0) -> np.ndarray:
    from aes_core import SBOX
    xor = (pt_col.astype(np.uint16) ^ key_guess) & 0xFF
    sbox_out = np.array(SBOX, dtype=np.uint8)[xor]
    return ((sbox_out >> bit) & 1).astype(bool)


def attack_byte(byte_idx: int, plaintexts: np.ndarray, traces: np.ndarray, bit: int = 0) -> dict:
    pt_col = plaintexts[:, byte_idx]
    T = traces.astype(np.float32)
    diff_matrix = np.zeros((256, T.shape[1]), dtype=np.float32)

    for k in range(256):
        sel = _selection_bit(pt_col, k, bit)
        if sel.sum() == 0 or (~sel).sum() == 0:
            continue
        diff_matrix[k] = np.abs(T[sel].mean(axis=0) - T[~sel].mean(axis=0))

    peak_per_guess = diff_matrix.max(axis=1)
    sorted_peaks   = np.sort(peak_per_guess)[::-1]
    best_guess     = int(peak_per_guess.argmax())
    confidence     = float(sorted_peaks[0] - sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0

    return {
        "best_guess":     best_guess,
        "max_diff":       float(peak_per_guess[best_guess]),
        "diff_matrix":    diff_matrix,
        "peak_per_guess": peak_per_guess,
        "confidence":     confidence,
    }


def recover_full_key(plaintexts: np.ndarray, traces: np.ndarray, bit: int = 0, verbose: bool = True) -> tuple:
    recovered, results = [], []
    for b in range(16):
        res = attack_byte(b, plaintexts, traces, bit=bit)
        recovered.append(res["best_guess"])
        results.append(res)
        if verbose:
            print(f"  Byte {b:2d}: 0x{res['best_guess']:02x}  "
                  f"peak_diff={res['max_diff']:.4f}  margin={res['confidence']:.4f}")
    return bytes(recovered), results
