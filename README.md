# DPA / CPA Side-Channel Attack Simulator

A research-grade side-channel attack simulator for AES-128 — pure Python + NumPy, no external crypto or attack libraries.

## Features

**Attacks**
- CPA — Correlation Power Analysis (vectorised Pearson, fastest)
- DPA — Differential Power Analysis (Kocher's original difference-of-means)
- MIA — Mutual Information Analysis (histogram-based KDE)
- Run all three at once with `--attack all`

**Leakage models**
- `hw` — Hamming Weight of SBOX output (default)
- `hd` — Hamming Distance (SBOX output vs previous state)
- `identity` — raw SBOX output value

**Countermeasures**
- Boolean masking (`--masking`)
- Byte-order shuffling (`--shuffling`)
- Clock jitter (`--jitter N`)
- Combine all three

**Analysis**
- SNR (Signal-to-Noise Ratio) per sample point
- TVLA — Welch's t-test leakage assessment (`--tvla`)
- Incremental CPA — correlation vs trace count curve (`--incremental`)
- Correlation heatmap — key guess × sample (`--heatmap`)
- Countermeasure benchmark — bytes recovered vs trace count (`--benchmark`)

**Output**
- All plots saved as PNG
- Self-contained HTML report (`--report`)
- Save/load trace sets as `.npz`

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic CPA (1000 traces, random key)
python main.py

# All three attacks
python main.py --attack all --traces 1000

# DPA with Hamming Distance model
python main.py --attack dpa --model hd

# Hard mode: masking + shuffling + jitter
python main.py --masking --shuffling --jitter 10 --traces 5000

# Full analysis with HTML report
python main.py --tvla --incremental --heatmap --report --outdir results/

# Supply your own key
python main.py --key deadbeefcafebabe0102030405060708

# Benchmark countermeasures
python main.py --benchmark --no-plot

# Save traces, reload later
python main.py --save-traces --outdir results/
python main.py --load-traces results/traces.npz
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--traces N` | 1000 | Number of power traces |
| `--seed S` | 42 | RNG seed |
| `--key HEX` | random | 32-char hex secret key |
| `--noise F` | 1.5 | Gaussian noise std dev |
| `--attack` | cpa | `cpa` / `dpa` / `mia` / `all` |
| `--model` | hw | `hw` / `hd` / `identity` |
| `--byte B` | 0 | Byte index for detail plots |
| `--jitter N` | 0 | Max clock-jitter offset (samples) |
| `--masking` | off | Boolean masking countermeasure |
| `--shuffling` | off | Byte-order shuffling countermeasure |
| `--tvla` | off | TVLA leakage assessment |
| `--incremental` | off | CPA correlation vs trace count |
| `--heatmap` | off | Correlation heatmap |
| `--benchmark` | off | Countermeasure benchmark |
| `--mia-step N` | 50 | MIA sample resolution (lower=slower/better) |
| `--outdir DIR` | results/ | Output directory |
| `--save-traces` | off | Save traces to `traces.npz` |
| `--load-traces F` | — | Load traces from `.npz` |
| `--report` | off | Generate HTML report |
| `--no-plot` | off | Skip all plots |

## Project structure

```
dpa_simulator/
├── aes_core.py          # Pure-Python AES-128 with intermediate value access
├── leakage_models.py    # HW / HD / identity leakage models
├── simulator.py         # Vectorised trace generation + SNR computation
├── countermeasures.py   # Countermeasure benchmark runner
├── plot.py              # All matplotlib figures
├── report.py            # Self-contained HTML report generator
├── main.py              # CLI entry point
├── cpa_attack.py        # Backward-compat shim → attacks/cpa.py
├── attacks/
│   ├── cpa.py           # CPA + incremental CPA
│   ├── dpa.py           # DPA (difference of means)
│   ├── mia.py           # MIA (mutual information)
│   └── tvla.py          # TVLA (Welch t-test)
└── requirements.txt
```
