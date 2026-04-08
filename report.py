import os
import base64
from datetime import datetime

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DPA Simulator Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1   {{ color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 8px; }}
  h2   {{ color: #3498db; margin-top: 32px; }}
  .meta {{ background: #1a1a2e; border-left: 4px solid #2ecc71; padding: 12px 16px; border-radius: 4px; font-family: monospace; font-size: 13px; }}
  .meta span {{ color: #2ecc71; }}
  .result {{ background: #1a1a2e; border-left: 4px solid {result_color}; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 15px; }}
  img  {{ max-width: 100%; border-radius: 6px; margin: 12px 0; box-shadow: 0 4px 12px #000; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: center; font-size: 13px; }}
  th {{ background: #2c3e50; color: #ecf0f1; }}
  .ok  {{ color: #2ecc71; font-weight: bold; }}
  .bad {{ color: #e74c3c; font-weight: bold; }}
  footer {{ margin-top: 40px; color: #555; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<h1>DPA / CPA Side-Channel Attack Report</h1>
<p style="color:#95a5a6">Generated: {timestamp}</p>
<div class="meta">
  <b>Secret key :</b> <span>{true_key}</span><br>
  <b>Recovered  :</b> <span>{recovered_key}</span><br>
  <b>Traces     :</b> {n_traces} &nbsp;|&nbsp;
  <b>Noise std  :</b> {noise_std} &nbsp;|&nbsp;
  <b>Jitter     :</b> {jitter} &nbsp;|&nbsp;
  <b>Seed       :</b> {seed}
</div>
<div class="result">
  Bytes correct: <b>{correct}/16</b> &nbsp;—&nbsp; {status}<br>
  Mean confidence margin: <b>{mean_conf:.4f}</b>
</div>
<h2>Byte-by-byte recovery</h2>
<table>
  <tr><th>Byte</th>{byte_headers}</tr>
  <tr><th>True</th>{true_row}</tr>
  <tr><th>Recovered</th>{rec_row}</tr>
  <tr><th>Match</th>{match_row}</tr>
</table>
{sections}
<footer>DPA Simulator &mdash; pure Python + NumPy</footer>
</body>
</html>
"""


def _img_tag(path: str) -> str:
    if not os.path.exists(path):
        return f"<p style='color:#e74c3c'>Image not found: {path}</p>"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{os.path.basename(path)}">'


def generate_report(
    outdir: str,
    true_key: bytes,
    recovered_key: bytes,
    results: list,
    n_traces: int,
    noise_std: float,
    jitter: int,
    seed: int,
    attack_type: str = "CPA",
    tvla_result: dict = None,
    benchmark_results: dict = None,
) -> str:
    correct   = sum(r == t for r, t in zip(recovered_key, true_key))
    status    = "FULL KEY RECOVERED" if correct == 16 else f"partial — {16 - correct} byte(s) wrong"
    mean_conf = sum(r.get("confidence", 0) for r in results) / 16

    byte_headers = "".join(f"<th>B{i}</th>" for i in range(16))
    true_row  = "".join(f"<td>0x{true_key[b]:02x}</td>" for b in range(16))
    rec_row   = "".join(
        f'<td class="{"ok" if recovered_key[b]==true_key[b] else "bad"}">0x{recovered_key[b]:02x}</td>'
        for b in range(16)
    )
    match_row = "".join(
        f'<td class="{"ok" if recovered_key[b]==true_key[b] else "bad"}">{"✓" if recovered_key[b]==true_key[b] else "✗"}</td>'
        for b in range(16)
    )

    sections = ""
    for fname, title in [
        ("traces.png",               "Power Traces"),
        ("snr.png",                  "Signal-to-Noise Ratio"),
        (f"cpa_byte0.png",           f"{attack_type} Detail — Byte 0"),
        ("cpa_heatmap_byte0.png",    "Correlation Heatmap — Byte 0"),
        ("key_recovery_summary.png", "Key Recovery Summary"),
        ("incremental_cpa_byte0.png","Incremental CPA — Byte 0"),
    ]:
        path = os.path.join(outdir, fname)
        if os.path.exists(path):
            sections += f"<h2>{title}</h2>\n{_img_tag(path)}\n"

    if tvla_result is not None:
        path = os.path.join(outdir, "tvla.png")
        leaky_str = "LEAKAGE DETECTED" if tvla_result["leaky"] else "No leakage detected"
        sections += f"<h2>TVLA — {leaky_str}</h2>\n{_img_tag(path)}\n"

    if benchmark_results is not None:
        path = os.path.join(outdir, "countermeasure_benchmark.png")
        sections += f"<h2>Countermeasure Benchmark</h2>\n{_img_tag(path)}\n"

    html = _HTML_TEMPLATE.format(
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        true_key      = true_key.hex(),
        recovered_key = recovered_key.hex(),
        n_traces      = n_traces,
        noise_std     = noise_std,
        jitter        = jitter,
        seed          = seed,
        correct       = correct,
        status        = status,
        mean_conf     = mean_conf,
        result_color  = "#2ecc71" if correct == 16 else "#e74c3c",
        byte_headers  = byte_headers,
        true_row      = true_row,
        rec_row       = rec_row,
        match_row     = match_row,
        sections      = sections,
    )

    out_path = os.path.join(outdir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path
