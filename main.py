"""Dilution ensemble trophy forecaster — gta6-trophy-rarity-forecast-01, variant C.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target.
Standard library only. Reads $MAKEFUN_INPUT_DIR/*.in (else /input), writes
$MAKEFUN_OUTPUT_DIR/<id>.out (else /output). Deterministic: no randomness,
no wall-clock dependence, sorted file order, fixed float formatting.

Model, in brief:
  * Counts, not rates. earners_i(t) = rate_i(t) * players(t) / 100. A rate is
    a ratio; when the player base explodes after launch, rates dilute even
    while earners grow. So players and earners are forecast separately and
    the rate is their ratio at the target.
  * Players: two growth laws fit on the history in log-time
    x = ln(1 + hours since launch 2026-11-19T00:00Z) — a log-linear law and
    a saturating law (log P = L - A*exp(-t/tau), tau picked from a grid by
    least squares) — averaged 50/50 in log space. If the history is flat
    (no player growth observed, e.g. every read pre-dates the first
    Exophase refresh), a launch-phase prior fires instead: players jump by
    the first-refresh multiple and then grow steadily in log space.
  * Earners: per-trophy log-earners slope against x, pooled per tier and
    shrunk hierarchically (trophy -> tier -> prior). The prior slope comes
    from an earners-vs-players exponent alpha_i: earners grow as
    players^alpha. alpha is 1 for a saturated trophy, high for the first
    story chapter and decaying with chapter number, near 1 for the
    most-earned trophy at launch, and a low floor otherwise — early
    trophies are earned by newcomers fast, late/grind trophies barely move.
  * Spread: per-trophy logit sigma built from a floor, the tier's residual
    spread, a horizon term growing with sqrt(days), and a staleness term
    when the tail of the history is flat. Rare trophies (predicted median
    under 0.1%) get a wider tail, scaled up with the horizon. Quantiles
    are median logit +/- z*sigma, mapped back to percent.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Launch-phase prior (tuned on the live-page backtest; see README) ---
# First Exophase refresh multiplies tracked players by ~6.06 (638 -> 3868);
# after that, log players grow ~0.0906/hour until the launch spike fades.
PRIOR_JUMP = 6.06
PRIOR_RATE = 0.0906
PRIOR_LAG_H = 1.5

# --- Earners-vs-players exponent prior ---
ALPHA_FLOOR = 0.13
ALPHA_CHAPTER = 0.85
ALPHA_DECAY = 0.40
ALPHA_MIN = 0.08
ALPHA_TOP = 0.98
SATURATED_RATE = 99.5

# --- Spread ---
SIG_FLOOR = 0.35
SIG_TIER_PRIOR = 0.15
SIG_HOR = 0.35
SIG_STALE = 0.20
RARE_CUT = 0.1
RARE_MULT = 0.15

# --- Player-law ensemble ---
LAW_W_A = 0.5
TAU_GRID_H = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)

CHAPTER_RE = re.compile(r"Complete Chapter (\d+)", re.IGNORECASE)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def round4(x):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return max(0.0, (dt - LAUNCH).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, sse) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return (a, b, sse)


def alpha_prior(trophy, last_rate, top_rate):
    """Earners-vs-players exponent for a trophy when history is uninformative."""
    if last_rate >= SATURATED_RATE:
        return 1.0
    m = CHAPTER_RE.search(trophy["description"])
    if m:
        n = int(m.group(1))
        return max(ALPHA_MIN, ALPHA_CHAPTER * (ALPHA_DECAY ** (n - 1)))
    desc = trophy["description"]
    if desc.startswith("Complete") and "final mission" in desc.lower():
        return max(ALPHA_MIN, ALPHA_CHAPTER * (ALPHA_DECAY ** 3))
    if last_rate == top_rate:
        return ALPHA_TOP
    return ALPHA_FLOOR


def player_cap_factor(target_age_h):
    """Max allowed player growth vs last read, by target age after launch.
    Launch day allows a large multiple; a week out, growth has flattened."""
    if target_age_h <= 48.0:
        return 30.0
    if target_age_h >= 120.0:
        return 4.0
    return 30.0 - (30.0 - 4.0) * (target_age_h - 48.0) / 72.0


def forecast_players(reads, t_last, t_target, p_last):
    """Return players at the target. Uses fitted laws when the history shows
    real player growth, else the launch-phase prior."""
    ts = [hours_since_launch(parse_time(r["at"])) for r in reads]
    ps = [float(r["players_tracked"]) for r in reads]
    distinct = len(set(ps))
    growth = (max(math.log(p) for p in ps) - min(math.log(p) for p in ps)
              if distinct > 1 else 0.0)
    x_target = math.log1p(t_target)
    if distinct >= 3 and growth > 0.05 and len(reads) >= 3:
        xs = [math.log1p(t) for t in ts]
        ys = [math.log(p) for p in ps]
        fit_a = ols(xs, ys)
        # Saturating law: log P = L - A*exp(-t/tau); try each tau, keep best.
        best_b = None
        for tau in TAU_GRID_H:
            us = [math.exp(-t / tau) for t in ts]
            fit = ols(us, ys)
            if fit is None:
                continue
            a_b, b_b, sse_b = fit
            # y = a_b + b_b*u with u = exp(-t/tau): b_b must be negative
            # (players rise toward the asymptote L = a_b).
            if b_b >= 0:
                continue
            if best_b is None or sse_b < best_b[0]:
                best_b = (sse_b, a_b, b_b, tau)
        preds = []
        if fit_a is not None:
            a_a, b_a, _ = fit_a
            preds.append(a_a + max(0.0, b_a) * x_target)
        if best_b is not None:
            _, a_b, b_b, tau = best_b
            preds.append(a_b + b_b * math.exp(-t_target / tau))
        if preds:
            return math.exp(sum(preds) / len(preds))
    # Flat history: launch-phase prior. The jump is the FIRST refresh's
    # catch-up (tracked players 638 -> 3868): it fires only if no such jump
    # is already in the history. If a jump was already observed, the
    # catch-up is done and only the steady post-refresh growth applies —
    # otherwise a second refresh read would wrongly forecast another x6.
    dh = max(0.0, t_target - t_last)
    jumped = any(ps[j + 1] / ps[j] > 1.5 for j in range(len(ps) - 1))
    if jumped:
        g = math.exp(PRIOR_RATE * dh)
    else:
        g = PRIOR_JUMP * math.exp(PRIOR_RATE * max(0.0, dh - PRIOR_LAG_H))
    return p_last * g


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    t_at = hours_since_launch(parse_time(case["at"]))
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    dx = max(0.0, math.log1p(t_target) - math.log1p(
        hours_since_launch(parse_time(history[-1]["at"]))))
    t_last = hours_since_launch(parse_time(history[-1]["at"]))

    last_rates = []
    for i in range(n_tr):
        r = None
        for h in reversed(history):
            if h["rates"][i] is not None:
                r = float(h["rates"][i])
                break
        last_rates.append(r if r is not None else 0.63)
    top_rate = max(r for r in last_rates if r < SATURATED_RATE)

    # --- Players ---
    p_last = float(history[-1]["players_tracked"])
    p_star = forecast_players(history, t_last, t_target, p_last)
    cap = p_last * player_cap_factor(t_target)
    p_star = min(max(p_star, p_last), cap)
    # Effective log-time slope of the player forecast: earners grow as
    # players^alpha, so the prior earners slope in x-space is alpha * b.
    b_player = math.log(p_star / p_last) / dx if dx > 1e-9 else 0.0
    # The history is informative about growth only if players actually
    # moved in it. A flat history (identical reads) makes every fitted
    # slope exactly zero with zero residual — a certain-looking zero that
    # would wrongly shrink the launch prior away. In that regime the
    # prior alone carries the forecast.
    p_hist = [float(h["players_tracked"]) for h in history]
    informative = (len(set(p_hist)) >= 3 and len(history) >= 3
                   and (max(math.log(p) for p in p_hist)
                        - min(math.log(p) for p in p_hist)) > 0.05)

    # --- Earners slopes: fit per trophy / per tier on log earners vs x ---
    # Only when the history is informative; otherwise the tier residuals
    # fall back to the prior spread and every slope is the alpha prior.
    xs = []
    for h in history:
        xs.append(math.log1p(hours_since_launch(parse_time(h["at"]))))
    x_varies = informative and (max(xs) - min(xs)) > 1e-9
    # log earners per trophy per read (None where the rate is missing)
    log_e = []
    for i in range(n_tr):
        row = []
        for j, h in enumerate(history):
            r = h["rates"][i]
            if r is None:
                row.append(None)
            else:
                e = max(float(r) * float(h["players_tracked"]) / 100.0, 0.5)
                row.append(math.log(e))
        log_e.append(row)

    def pooled_slope(idxs):
        """Pooled OLS slope of log earners on x, demeaned per trophy."""
        num = 0.0
        den = 0.0
        for i in idxs:
            pts = [(xs[j], log_e[i][j]) for j in range(len(xs))
                   if log_e[i][j] is not None]
            if len(pts) < 2:
                continue
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            for x, y in pts:
                num += (x - mx) * (y - my)
                den += (x - mx) ** 2
        if den <= 1e-12:
            return None, 0
        return num / den, sum(1 for i in idxs for v in log_e[i] if v is not None)

    tiers = {}
    for i, t in enumerate(trophies):
        tiers.setdefault(t["tier"], []).append(i)
    tier_slope = {}
    tier_resid = {}
    all_idx = list(range(n_tr))
    g_slope, _ = pooled_slope(all_idx)
    if g_slope is None:
        g_slope = 0.0
    for tier, idxs in tiers.items():
        if not informative:
            tier_slope[tier] = g_slope
            tier_resid[tier] = SIG_TIER_PRIOR
            continue
        s, cnt = pooled_slope(idxs)
        if s is None:
            tier_slope[tier] = g_slope
            tier_resid[tier] = SIG_TIER_PRIOR
            continue
        # Shrink the tier slope toward the global slope (8 pseudo-obs).
        tier_slope[tier] = (cnt * s + 8.0 * g_slope) / (cnt + 8.0)
        # Residual sd of the tier fit around its own line.
        sq = 0.0
        k = 0
        for i in idxs:
            pts = [(xs[j], log_e[i][j]) for j in range(len(xs))
                   if log_e[i][j] is not None]
            if len(pts) < 2:
                continue
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            for x, y in pts:
                sq += (y - (my + s * (x - mx))) ** 2
                k += 1
        tier_resid[tier] = math.sqrt(sq / k) if k > 2 else SIG_TIER_PRIOR

    # --- Staleness: how long the tail of the history has been flat ---
    last_change_h = None
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if prev is not None and key != prev:
            last_change_h = hours_since_launch(parse_time(h["at"]))
        prev = key
    if last_change_h is None:
        last_change_h = hours_since_launch(parse_time(history[0]["at"]))
    stale_days = max(0.0, (t_at - last_change_h) / 24.0)
    sig_stale = min(0.5, SIG_STALE * math.sqrt(stale_days))

    # --- Per-trophy forecast ---
    rows = []
    for i, t in enumerate(trophies):
        a_i = alpha_prior(t, last_rates[i], top_rate)
        s_prior = a_i * b_player
        # Per-trophy fitted slope, if the history identifies one.
        s_fit = None
        n_i = 0
        if x_varies:
            pts = [(xs[j], log_e[i][j]) for j in range(len(xs))
                   if log_e[i][j] is not None]
            n_i = len(pts)
            if n_i >= 3:
                fit = ols([p[0] for p in pts], [p[1] for p in pts])
                if fit is not None:
                    s_fit = fit[1]
        if s_fit is None:
            slope = s_prior
        else:
            # Precision-weighted: fitted slope, tier slope, prior slope.
            slope = (n_i * s_fit + 3.0 * tier_slope[t["tier"]]
                     + 3.0 * s_prior) / (n_i + 6.0)
        slope = max(0.0, slope)
        e_last = max(last_rates[i] * p_last / 100.0, 0.5)
        e_star = e_last * math.exp(slope * dx)
        med = clamp_pct(100.0 * e_star / p_star)
        # Spread: floor + tier residual + horizon growth + staleness.
        sig_tier = tier_resid.get(t["tier"], SIG_TIER_PRIOR)
        sigma = math.sqrt(SIG_FLOOR ** 2 + sig_tier ** 2
                          + (SIG_HOR * math.sqrt(horizon_days)) ** 2
                          + sig_stale ** 2)
        if med < RARE_CUT:
            sigma *= 1.0 + RARE_MULT * min(1.0, horizon_days / 4.5)
        lg = logit(med)
        qs = []
        for z in Z:
            v = 100.0 / (1.0 + math.exp(-(lg + z * sigma)))
            qs.append(round4(v))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
