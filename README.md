# Roblox Up-and-Coming Forecast

This Node.js 22 entry reads `task.json` and every snapshot in `history.ndjson` from `MAKEFUN_INPUT_DIR` (default `/input`) and writes a deterministic `forecast.json` to `MAKEFUN_OUTPUT_DIR` (default `/output`). It uses only Node.js built-ins and does not access the network.

For each game's observations at ranks 1–24, it sums Gaussian rank kernels for each scored place. Observation weights decay with a 21-hour half-life, and the final read receives a 1.5× boost. Games unseen for more than 12 hours receive an additional age decay. When creation details are available, games that age out of the chart during the task window are downweighted by the fraction of that window for which they remain eligible.

Each place's game scores are normalized to 0.84, rounded to six decimal places, and capped at 100 rows. The remaining probability is intentionally unnamed. The ordered `top10` is selected by each game's probability-weighted expected place, with total probability mass and universe ID as tie-breakers. The task window and number of places are read from `task.json`.

Run locally with:

```sh
MAKEFUN_INPUT_DIR=./kit/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
