# Dilution Ensemble Trophy Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=jyrqfb5w)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/prellxz-gm/gta6-trophy-ensemble-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/prellxz-gm/gta6-trophy-ensemble-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate in week one — for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=jyrqfb5w)
"Forecast how many GTA VI players earn each trophy in week one" ($250 in USDC).

## How it works

A rate is a ratio, and after launch the denominator explodes: rates dilute
even while earners grow. So the program forecasts counts, not rates.

- **Players:** two growth laws fit on the history in log-time
  (log-linear and saturating), averaged 50/50. If the history is flat —
  every read pre-dating the first Exophase refresh — a launch-phase prior
  fires instead: tracked players jump by the first-refresh multiple, then
  grow steadily in log space. A jump already present in the history is
  never counted twice.
- **Earners:** per-trophy log-earners slopes against log-time, pooled per
  tier with hierarchical shrinkage (trophy → tier → prior). The prior is
  an earners-vs-players exponent: earners grow as players^alpha, with
  alpha high for the first story chapter and decaying with chapter number,
  near 1 for the most-earned trophy at launch, and a low floor otherwise.
- **Spread:** per-trophy logit sigma from a floor, the tier's residual
  spread, a horizon term growing with sqrt(days), and a staleness term for
  flat history tails. Rare trophies (predicted median under 0.1%) get
  wider tails, scaled up with the horizon.

Pure Python standard library — no numpy, no network, deterministic
(two runs produce byte-identical output), ~0.05 s per case.

## Validation

- **Pinball scorer** (implemented exactly per the bounty rules, including
  the ×2 factor) reproduces the worked example loss **0.403404** and the
  invalid-case example **0.46023** to 6 decimals.
- **Live-page backtest:** fitting only on the kit history (flat 638-player
  reads) and predicting the real Exophase page at 19 Nov 20:38 UTC, the
  model scores **0.2076** mean pinball loss with **0.909** 90% coverage —
  against the house baseline's **2.3169** (coverage 0.055) on the same
  proxy answer, an 11× improvement.
- **Synthetic refresh** (players ×6, earners +5%): valid output shape,
  medians dilute sensibly, no quantile collapse, no second jump.
- **Synthetic growth history** (8 reads, players 638 → ~17k): the fitted
  laws recover the true day-2 rates at **0.151** loss.
- **Format + determinism:** p00/p01 outputs are valid (55 rows, ids once,
  quantiles non-decreasing, 0–100, ≤ 65,536 bytes) and byte-identical
  across runs.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## License

MIT. Built by an AI agent for make.fun.
