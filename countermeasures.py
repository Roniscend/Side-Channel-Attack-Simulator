import numpy as np

COUNTERMEASURES = {
    "none":      {"masking": False, "shuffling": False, "jitter": 0},
    "jitter":    {"masking": False, "shuffling": False, "jitter": 10},
    "shuffling": {"masking": False, "shuffling": True,  "jitter": 0},
    "masking":   {"masking": True,  "shuffling": False, "jitter": 0},
    "all":       {"masking": True,  "shuffling": True,  "jitter": 10},
}


def benchmark_countermeasure(
    key: bytes,
    countermeasure: str = "none",
    trace_counts: list = None,
    noise_std: float = 1.5,
    seed: int = 42,
) -> dict:
    from simulator import generate_traces
    from attacks.cpa import recover_full_key

    if trace_counts is None:
        trace_counts = [100, 250, 500, 1000, 2000, 5000]

    cm = COUNTERMEASURES.get(countermeasure, COUNTERMEASURES["none"])
    results = {}

    for n in trace_counts:
        pts, traces = generate_traces(key, n, seed=seed, noise_std=noise_std, verbose=False, **cm)
        recovered, _ = recover_full_key(pts, traces, verbose=False)
        correct = sum(r == t for r, t in zip(recovered, key))
        results[n] = correct
        print(f"  [{countermeasure:10s}] {n:5d} traces → {correct:2d}/16 bytes correct")

    return results
