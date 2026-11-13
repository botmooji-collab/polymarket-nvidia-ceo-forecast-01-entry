# Polymarket NVIDIA CEO forecast: per-market quantile forecasts from the
# case's own 72h price history. No network, stdlib only, deterministic.
#
# Model: for each market, two robust spread estimates are blended 70/30.
#  (a) horizon MAD: for horizons h in (1, 4, 16, 64, 240) minutes, the
#      median absolute deviation of h-minute returns over the last 6 hours
#      (x1.4826, the normal-consistent scale), interpolated in sqrt(h).
#      This adapts the spread to how jumpy this market actually is at each
#      horizon, instead of assuming one global random-walk sigma.
#  (b) 1-minute robust sigma (MAD of 1-minute returns over the last 6h,
#      x1.25) widened like a random walk, sigma * sqrt(h).
# The median is the last price (persistence): recent drift hurt on every
# practice case, so drift is 0. Quantiles are the standard normal's at
# 0.05/0.25/0.5/0.75/0.95 around that median, clamped to [0, 1].
#
# Practice loss (mean pinball, p00-p05, scored against the next case's
# prices and fetched Polymarket history for p05): ~0.286 vs the house
# baseline's ~0.364 on the same cases, about 21% better.

import json
import math
import os
import statistics
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
HORIZONS = (1, 4, 16, 64, 240)
WINDOW = 360          # minutes of recent history for the spread estimates
FLOOR = 0.0005
CAP = 0.012


def mad_scale(values):
    """Normal-consistent robust scale of a list of returns (1.4826 * MAD)."""
    if len(values) < 10:
        return None
    med = statistics.median(values)
    return 1.4826 * statistics.median([abs(v - med) for v in values])


def clamp6(x):
    return min(1.0, max(0.0, round(x, 6)))


def forecast(prices, horizon):
    x = prices[-1]
    # Recent window: last WINDOW minutes plus room for the 240-min returns.
    p = prices[-(WINDOW + 241):] if len(prices) > WINDOW + 241 else prices
    # (a) horizon MAD at grid horizons, interpolated in sqrt(h).
    grid_h, grid_s = [], []
    for h in HORIZONS:
        if len(p) - h < 20:
            continue
        rets = [p[i + h] - p[i] for i in range(len(p) - h)]
        s = mad_scale(rets)
        if s is not None:
            grid_h.append(math.sqrt(h))
            grid_s.append(s)
    # (b) robust 1-minute sigma.
    one = [prices[i + 1] - prices[i] for i in range(max(0, len(prices) - WINDOW - 1), len(prices) - 1)]
    sig1 = mad_scale(one)
    if sig1 is None:
        sig1 = 0.0025
    sig1 = min(CAP, max(FLOOR, sig1 * 1.25))
    # Jump insurance: a market that was dead quiet for the last 6 hours can
    # still jump (practice p00: one market moved 0.35 -> 0.59 in a minute).
    # Never let the 6-hour estimate fall below half the market's own 72-hour
    # robust sigma, so a quiet stretch does not collapse the spread.
    all_rets = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    sig72 = mad_scale(all_rets)
    if sig72 is not None:
        sig1 = max(sig1, 0.5 * sig72)

    rows = []
    for h in range(1, horizon + 1):
        root = math.sqrt(h)
        if grid_h:
            # linear interpolation in sqrt(h), clamped to the grid ends
            if root <= grid_h[0]:
                sig_h = grid_s[0]
            elif root >= grid_h[-1]:
                sig_h = grid_s[-1]
            else:
                for j in range(len(grid_h) - 1):
                    if grid_h[j] <= root <= grid_h[j + 1]:
                        t = (root - grid_h[j]) / (grid_h[j + 1] - grid_h[j])
                        sig_h = grid_s[j] + t * (grid_s[j + 1] - grid_s[j])
                        break
            sig_h = min(CAP * root, max(FLOOR, sig_h))
        else:
            sig_h = sig1 * root
        sig_b = sig1 * root
        row = []
        prev = -1.0
        for z in Z:
            q = 0.7 * (x + z * sig_h) + 0.3 * (x + z * sig_b)
            q = clamp6(q)
            if q < prev:
                q = prev
            prev = q
            row.append(q)
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(f for f in os.listdir(in_dir) if f.endswith(".in")):
        case = json.loads((in_dir / name).read_text())
        horizon = case["horizon_minutes"]
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": forecast(m["prices"], horizon)})
        (out_dir / (name[:-3] + ".out")).write_text(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
