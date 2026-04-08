import numpy as np
from aes_core import SBOX, hamming_weight

_HW      = np.array([bin(x).count('1') for x in range(256)], dtype=np.float32)
_HW_SBOX = _HW[np.array(SBOX, dtype=np.uint8)]
_ID_SBOX = np.array(SBOX, dtype=np.float32)


def hw_model(pt_col: np.ndarray, key_guess: int) -> np.ndarray:
    xor = (pt_col.astype(np.uint16) ^ key_guess) & 0xFF
    return _HW_SBOX[xor]


def hd_model(pt_col: np.ndarray, key_guess: int, prev: np.ndarray = None) -> np.ndarray:
    xor = (pt_col.astype(np.uint16) ^ key_guess) & 0xFF
    sbox_out = np.array(SBOX, dtype=np.uint8)[xor]
    prev_state = xor.astype(np.uint8) if prev is None else prev.astype(np.uint8)
    diff = (sbox_out ^ prev_state).astype(np.uint8)
    return _HW[diff]


def identity_model(pt_col: np.ndarray, key_guess: int) -> np.ndarray:
    xor = (pt_col.astype(np.uint16) ^ key_guess) & 0xFF
    return _ID_SBOX[xor]


def bit_model(pt_col: np.ndarray, key_guess: int, bit: int = 0) -> np.ndarray:
    xor = (pt_col.astype(np.uint16) ^ key_guess) & 0xFF
    sbox_out = np.array(SBOX, dtype=np.uint8)[xor]
    return ((sbox_out >> bit) & 1).astype(np.float32)


MODELS = {
    "hw":       hw_model,
    "hd":       hd_model,
    "identity": identity_model,
}
