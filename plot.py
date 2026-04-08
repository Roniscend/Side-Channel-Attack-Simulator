import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

C_CORRECT = "#2ecc71"
C_WRONG   = "#e74c3c"
C_NEUTRAL = "#3498db"
C_GRAY    = "#95a5a6"


def plot_traces(traces: np.ndarray, n: int = 8, title: str = "Simulated Power Traces"):
    fig, ax = plt.subplots(figsize=(13, 4))
    cmap = plt.cm.viridis
    for i in range(min(n, len(traces))):
        ax.plot(traces[i], alpha=0.65, linewidth=0.6,
                color=cmap(i / max(n - 1, 1)), label=f"Trace {i}")
    ax.plot(traces[:n].mean(axis=0), color="white", linewidth=1.4, linestyle="--", label="Mean", zorder=5)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.set_title(title, color="white")
    ax.set_xlabel("Sample index", color=C_GRAY)
    ax.set_ylabel("Power (a.u.)", color=C_GRAY)
    ax.tick_params(colors=C_GRAY)
    ax.legend(fontsize=7, ncol=4, facecolor="#0f3460", labelcolor="white")
    plt.tight_layout()
    return fig


def plot_snr(snr: np.ndarray, title: str = "Signal-to-Noise Ratio"):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.fill_between(range(len(snr)), snr, alpha=0.4, color=C_NEUTRAL)
    ax.plot(snr, linewidth=0.8, color=C_NEUTRAL)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("SNR")
    plt.tight_layout()
    return fig


def plot_cpa_result(byte_idx: int, result: dict, true_key_byte: int):
    corr       = result["corr_matrix"]
    best       = result["best_guess"]
    peak       = result["peak_per_guess"]
    confidence = result.get("confidence", 0.0)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 1, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    wrong = (true_key_byte + 1) % 256
    for k in range(256):
        if k not in (true_key_byte, wrong):
            ax1.plot(corr[k], color=C_GRAY, linewidth=0.3, alpha=0.15)
    ax1.plot(corr[wrong],         color=C_WRONG,   linewidth=0.8, alpha=0.8, label=f"Wrong  0x{wrong:02x}")
    ax1.plot(corr[true_key_byte], color=C_CORRECT, linewidth=1.2,            label=f"Correct 0x{true_key_byte:02x}")
    ax1.set_title(f"CPA — Byte {byte_idx}: |Pearson r| over samples")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("|r|")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    colors = [C_CORRECT if i == true_key_byte else
              (C_WRONG if i == best and best != true_key_byte else C_NEUTRAL)
              for i in range(256)]
    ax2.bar(range(256), peak, color=colors, width=1.0, alpha=0.85)
    ax2.axvline(true_key_byte, color=C_CORRECT, linestyle="--", linewidth=1.3, label=f"True  0x{true_key_byte:02x}")
    if best != true_key_byte:
        ax2.axvline(best, color=C_WRONG, linestyle="--", linewidth=1.3, label=f"Best  0x{best:02x}")
    ax2.set_title(f"Peak |r| per key guess — Byte {byte_idx}  (margin: {confidence:.4f})")
    ax2.set_xlabel("Key guess (0–255)")
    ax2.set_ylabel("Peak |r|")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(result: dict, byte_idx: int, true_key_byte: int):
    corr = result["corr_matrix"]
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(corr, aspect="auto", origin="lower", cmap="inferno",
                   interpolation="nearest", vmin=0, vmax=corr.max())
    ax.axhline(true_key_byte, color=C_CORRECT, linewidth=1.0, linestyle="--",
               label=f"True key 0x{true_key_byte:02x}")
    ax.set_title(f"CPA Correlation Heatmap — Byte {byte_idx}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Key guess")
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="|Pearson r|")
    plt.tight_layout()
    return fig


def plot_key_recovery_summary(recovered: bytes, true_key: bytes, results: list):
    n = 16
    x = np.arange(n)
    rec_peaks  = [results[b]["peak_per_guess"][recovered[b]] for b in range(n)]
    true_peaks = [results[b]["peak_per_guess"][true_key[b]]  for b in range(n)]
    confs      = [results[b].get("confidence", 0.0) for b in range(n)]
    correct    = [recovered[b] == true_key[b] for b in range(n)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"hspace": 0.55})

    bar_colors = [C_CORRECT if c else C_WRONG for c in correct]
    axes[0].bar(x - 0.2, true_peaks, 0.35, label="True key",      color=C_CORRECT, alpha=0.7)
    axes[0].bar(x + 0.2, rec_peaks,  0.35, label="Recovered key", color=bar_colors, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"B{i}" for i in range(n)], fontsize=8)
    axes[0].set_ylabel("Peak |r|")
    axes[0].set_title(f"Key Recovery — {sum(correct)}/16 bytes correct")
    axes[0].legend()

    axes[1].bar(x, confs, color=bar_colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"B{i}" for i in range(n)], fontsize=8)
    axes[1].set_ylabel("Margin")
    axes[1].set_title("Distinguishing margin per byte")

    axes[2].axis("off")
    col_labels = [f"B{i}" for i in range(n)]
    true_row   = [f"0x{true_key[b]:02x}" for b in range(n)]
    rec_row    = [f"0x{recovered[b]:02x}" for b in range(n)]
    match_row  = ["✓" if correct[b] else "✗" for b in range(n)]
    tbl = axes[2].table(
        cellText=[true_row, rec_row, match_row],
        rowLabels=["True", "Recovered", "Match"],
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white")
        elif row == 3:
            val = match_row[col - 1] if col > 0 else ""
            cell.set_facecolor(C_CORRECT if val == "✓" else (C_WRONG if val == "✗" else "#2c3e50"))
            cell.set_text_props(color="white")
    axes[2].set_title("Byte-by-byte comparison", pad=12)

    plt.tight_layout()
    return fig


def plot_incremental_cpa(trace_counts: np.ndarray, peak_over_time: np.ndarray, true_key_byte: int, byte_idx: int):
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in range(256):
        if k == true_key_byte:
            continue
        ax.plot(trace_counts, peak_over_time[:, k], color=C_GRAY, linewidth=0.4, alpha=0.3)
    ax.plot(trace_counts, peak_over_time[:, true_key_byte],
            color=C_CORRECT, linewidth=2.0, label=f"Correct 0x{true_key_byte:02x}")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=1.0, label="|r|=0.5 guide")
    ax.set_title(f"Incremental CPA — Byte {byte_idx}: correlation vs trace count")
    ax.set_xlabel("Number of traces")
    ax.set_ylabel("Peak |r|")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_tvla(tvla_result: dict):
    t   = tvla_result["t_stat"]
    thr = tvla_result["threshold"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    ax.plot(t, linewidth=0.7, color=C_NEUTRAL)
    ax.axhline( thr, color=C_WRONG, linestyle="--", linewidth=1.2, label=f"+{thr}")
    ax.axhline(-thr, color=C_WRONG, linestyle="--", linewidth=1.2, label=f"-{thr}")
    ax.fill_between(range(len(t)), t, 0, where=np.abs(t) > thr, color=C_WRONG, alpha=0.3, label="Leaky")
    ax.set_title(f"TVLA — Welch t-test  ({'LEAKAGE DETECTED' if tvla_result['leaky'] else 'No leakage detected'})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("t-statistic")
    ax.legend(fontsize=8)

    axes[1].plot(tvla_result["fixed_mean"],  color=C_CORRECT, linewidth=0.8, label="Fixed set mean")
    axes[1].plot(tvla_result["random_mean"], color=C_NEUTRAL,  linewidth=0.8, label="Random set mean")
    axes[1].set_title("Mean power trace: fixed vs random plaintexts")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Power (a.u.)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_countermeasure_benchmark(benchmark_results: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.tab10
    for idx, (cm_name, data) in enumerate(benchmark_results.items()):
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=cm_name,
                color=cmap(idx / max(len(benchmark_results) - 1, 1)))
    ax.axhline(16, color=C_CORRECT, linestyle=":", linewidth=1.0, label="Full key (16/16)")
    ax.set_title("Countermeasure Benchmark — bytes recovered vs trace count")
    ax.set_xlabel("Number of traces")
    ax.set_ylabel("Bytes correctly recovered")
    ax.set_ylim(-0.5, 17)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig
