import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from simulator import generate_traces, compute_snr
from plot import (
    plot_traces, plot_snr, plot_cpa_result, plot_correlation_heatmap,
    plot_key_recovery_summary, plot_incremental_cpa, plot_tvla,
    plot_countermeasure_benchmark,
)
from report import generate_report


def _save(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _hex_key(s):
    s = s.strip()
    if len(s) != 32 or not all(c in "0123456789abcdefABCDEF" for c in s):
        raise argparse.ArgumentTypeError("Key must be a 32-character hex string")
    return bytes.fromhex(s)


def parse_args():
    p = argparse.ArgumentParser(description="DPA/CPA Side-Channel Attack Simulator")
    p.add_argument("--traces",      type=int,      default=1000)
    p.add_argument("--seed",        type=int,      default=42)
    p.add_argument("--key",         type=_hex_key, default=None)
    p.add_argument("--noise",       type=float,    default=1.5)
    p.add_argument("--attack",      choices=["cpa","dpa","mia","all"], default="cpa")
    p.add_argument("--mia-step",    type=int,      default=50)
    p.add_argument("--model",       choices=["hw","hd","identity"], default="hw")
    p.add_argument("--byte",        type=int,      default=0)
    p.add_argument("--jitter",      type=int,      default=0)
    p.add_argument("--masking",     action="store_true")
    p.add_argument("--shuffling",   action="store_true")
    p.add_argument("--tvla",        action="store_true")
    p.add_argument("--benchmark",   action="store_true")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--heatmap",     action="store_true")
    p.add_argument("--outdir",      type=str,      default="results")
    p.add_argument("--save-traces", action="store_true")
    p.add_argument("--load-traces", type=str,      default=None)
    p.add_argument("--report",      action="store_true")
    p.add_argument("--no-plot",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.traces < 10:
        print("Error: --traces must be >= 10"); sys.exit(1)
    if not (0 <= args.byte <= 15):
        print("Error: --byte must be 0–15"); sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if args.key:
        secret_key = args.key
    else:
        rng = np.random.default_rng(args.seed)
        secret_key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))

    _banner(secret_key, args)

    if args.load_traces:
        print(f"[1] Loading traces from {args.load_traces}...")
        data = np.load(args.load_traces)
        plaintexts, traces = data["plaintexts"], data["traces"]
        print(f"    Loaded {traces.shape[0]} traces  shape={traces.shape}")
    else:
        print("[1] Generating power traces...")
        t0 = time.time()
        plaintexts, traces = generate_traces(
            secret_key, args.traces,
            seed=args.seed + 1,
            noise_std=args.noise,
            jitter=args.jitter,
            masking=args.masking,
            shuffling=args.shuffling,
        )
        print(f"    Done in {time.time()-t0:.2f}s  shape={traces.shape}")

        if args.save_traces:
            p = os.path.join(args.outdir, "traces.npz")
            np.savez_compressed(p, plaintexts=plaintexts, traces=traces)
            print(f"    Saved → {p}")
    print()

    print("[2] Computing SNR...")
    snr = compute_snr(plaintexts, traces, secret_key)
    print(f"    Peak SNR = {float(snr.max()):.3f}  at sample {int(snr.argmax())}")
    print()

    attack_modes = ["cpa", "dpa", "mia"] if args.attack == "all" else [args.attack]
    all_attack_results = {}

    for mode in attack_modes:
        print(f"[3] Running {mode.upper()} attack (model={args.model})...")
        t0 = time.time()

        if mode == "cpa":
            from attacks.cpa import recover_full_key
            recovered, results = recover_full_key(plaintexts, traces, model=args.model, verbose=True)
        elif mode == "dpa":
            from attacks.dpa import recover_full_key
            recovered, results = recover_full_key(plaintexts, traces, verbose=True)
        elif mode == "mia":
            from attacks.mia import recover_full_key
            mia_step = getattr(args, "mia_step", 50)
            print(f"    (MIA: sample_step={mia_step})")
            recovered, results = recover_full_key(plaintexts, traces, model=args.model, sample_step=mia_step, verbose=True)

        elapsed = time.time() - t0
        correct = sum(r == t for r, t in zip(recovered, secret_key))
        mean_conf = np.mean([r.get("confidence", 0) for r in results])

        print(f"\n    Finished in {elapsed:.2f}s")
        print(f"    True key      : {secret_key.hex()}")
        print(f"    Recovered key : {recovered.hex()}")
        print(f"    Bytes correct : {correct}/16  ({'FULL KEY RECOVERED' if correct==16 else 'partial'})")
        print(f"    Mean margin   : {mean_conf:.4f}")
        print()

        all_attack_results[mode] = (recovered, results, correct)

    primary_mode = attack_modes[0]
    recovered_key, results, correct = all_attack_results[primary_mode]

    if correct < 16:
        print("  Tip: try --traces 3000, or lower --noise / --jitter")
        print()

    tvla_result = None
    if args.tvla:
        print("[4] Running TVLA leakage assessment...")
        from attacks.tvla import run_tvla
        tvla_result = run_tvla(
            secret_key, n_traces=min(args.traces, 2000),
            seed=args.seed + 10, noise_std=args.noise,
            jitter=args.jitter, masking=args.masking, shuffling=args.shuffling,
        )
        status = "LEAKAGE DETECTED" if tvla_result["leaky"] else "No leakage detected"
        print(f"    {status}  ({int(tvla_result['leaky_mask'].sum())} leaky sample points)")
        print()

    benchmark_results = None
    if args.benchmark:
        print("[5] Benchmarking countermeasures...")
        from countermeasures import benchmark_countermeasure, COUNTERMEASURES
        benchmark_results = {}
        for cm_name in COUNTERMEASURES:
            print(f"\n  -- {cm_name} --")
            benchmark_results[cm_name] = benchmark_countermeasure(
                secret_key, cm_name,
                trace_counts=[100, 250, 500, 1000, 2000],
                noise_std=args.noise, seed=args.seed,
            )
        print()

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("[6] Generating plots...")

        _save(plot_traces(traces, n=8),                os.path.join(args.outdir, "traces.png"))
        _save(plot_snr(snr),                           os.path.join(args.outdir, "snr.png"))
        _save(plot_cpa_result(args.byte, results[args.byte], secret_key[args.byte]),
              os.path.join(args.outdir, f"cpa_byte{args.byte}.png"))
        _save(plot_key_recovery_summary(recovered_key, secret_key, results),
              os.path.join(args.outdir, "key_recovery_summary.png"))

        if args.heatmap:
            _save(plot_correlation_heatmap(results[args.byte], args.byte, secret_key[args.byte]),
                  os.path.join(args.outdir, f"cpa_heatmap_byte{args.byte}.png"))

        if args.incremental and primary_mode == "cpa":
            from attacks.cpa import incremental_cpa
            peak_ot, counts = incremental_cpa(args.byte, plaintexts, traces, steps=30, model=args.model)
            _save(plot_incremental_cpa(counts, peak_ot, secret_key[args.byte], args.byte),
                  os.path.join(args.outdir, f"incremental_cpa_byte{args.byte}.png"))

        if tvla_result is not None:
            _save(plot_tvla(tvla_result), os.path.join(args.outdir, "tvla.png"))

        if benchmark_results is not None:
            _save(plot_countermeasure_benchmark(benchmark_results),
                  os.path.join(args.outdir, "countermeasure_benchmark.png"))

        print(f"    Plots saved to {args.outdir}/")
        print()

    if args.report:
        print("[7] Generating HTML report...")
        rpath = generate_report(
            outdir=args.outdir,
            true_key=secret_key,
            recovered_key=recovered_key,
            results=results,
            n_traces=args.traces,
            noise_std=args.noise,
            jitter=args.jitter,
            seed=args.seed,
            attack_type=primary_mode.upper(),
            tvla_result=tvla_result,
            benchmark_results=benchmark_results,
        )
        print(f"    Report → {rpath}")
        print()

    print("Done.")


def _banner(key, args):
    cm_flags = []
    if args.masking:   cm_flags.append("masking")
    if args.shuffling: cm_flags.append("shuffling")
    if args.jitter:    cm_flags.append(f"jitter={args.jitter}")
    cm_str = ", ".join(cm_flags) if cm_flags else "none"

    print("=" * 62)
    print("  DPA / CPA Side-Channel Attack Simulator")
    print("=" * 62)
    print(f"  Secret key     : {key.hex()}")
    print(f"  Traces         : {args.traces}")
    print(f"  Noise std      : {args.noise}")
    print(f"  Attack         : {args.attack.upper()}")
    print(f"  Leakage model  : {args.model}")
    print(f"  Countermeasures: {cm_str}")
    print(f"  Output dir     : {args.outdir}/")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
