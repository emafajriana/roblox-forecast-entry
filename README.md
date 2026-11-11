# Roblox Up-and-Coming Forecast

This Node.js 22 entry reads `task.json` and every snapshot in `history.ndjson` from `MAKEFUN_INPUT_DIR` (default `/input`) and writes a deterministic `forecast.json` to `MAKEFUN_OUTPUT_DIR` (default `/output`). It uses only Node.js built-ins and does not access the network.

The forecast is an equal blend of two estimates. The rank-kernel estimate uses top-25 observations with an 18-hour half-life, doubles the last read's observations, and decays games by the time since their last appearance. The Monte Carlo estimate uses recent log-rank observations to estimate global volatility and per-game drift, then simulates 800 deterministic 72-hour chart paths using Mulberry32 and Box–Muller Gaussian samples.

If `details.json` is present, both estimates account for games aging out 28 days after creation. Final probabilities are rounded to six decimals, values below 0.0005 are omitted, and each place has at most 100 named games; remaining probability stays unnamed. The top 10 is ordered by the sum of probability-weighted place values. Inputs and output paths are configurable with `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR`.

Run locally with:

```sh
MAKEFUN_INPUT_DIR=./kit/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
