"""
Dr Praise Gold Bank — Pine-aligned backtest on XAUUSD M1 (HistData CSV).

NY OR 09:30–09:35 (inclusive minutes), filters, Instant/Retracement, SL types,
second chance, 16:00 session close. Market fills next bar open. Iterates only
~09:25–16:30 NY per day for speed.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import time as dt_time
from typing import Any, Literal, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = "/Users/mac/Desktop/fl/orb_alert_bot/backtest_data/XAUUSD 2009-2026 DATA"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

Direction = Literal["long", "short"]


@dataclass
class Params:
    contracts: int = 1
    risk_multiplier: float = 5.0
    sl_points: int = 2
    entry_type: str = "Instant"
    retracement_percent: float = 50.0
    sl_type: str = "Opposite Range"
    enable_second_chance: bool = True
    use_wick_filter: bool = True
    max_wick_percent: float = 50.0
    use_breakout_distance_filter: bool = True
    min_breakout_multiplier: float = 0.1
    max_breakout_multiplier: float = 1.6
    use_trailing_sl: bool = False
    profit_r_multiplier: float = 1.0
    atr_length: int = 14
    atr_multiplier: float = 1.0
    tick_size: float = 0.01
    force_session_close: bool = True
    session_end_hour: int = 16
    session_end_minute: int = 0
    or_start_hour: int = 9
    or_start_minute: int = 30
    or_end_minute: int = 35
    trade_monday: bool = True
    trade_tuesday: bool = True
    trade_wednesday: bool = True
    trade_thursday: bool = True
    trade_friday: bool = True
    initial_capital: float = 50_000.0
    dollar_per_point: float = 1.0
    data_tz: str = "UTC"


def load_all_m1_csv(data_root: str) -> pd.DataFrame:
    pattern = os.path.join(data_root, "**", "DAT_ASCII_XAUUSD_M1*.csv")
    files = sorted({os.path.realpath(p) for p in glob.glob(pattern, recursive=True)})
    if not files:
        raise RuntimeError(f"No DAT_ASCII_XAUUSD_M1*.csv under {data_root!r}")
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, sep=";", header=None)
            df.columns = ["dt", "open", "high", "low", "close", "volume"]
            df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
            frames.append(df)
            print(f"  [OK] {os.path.basename(path)} — {len(df):,} rows")
        except Exception as e:
            print(f"  [ERR] {path}: {e}")
    if not frames:
        raise RuntimeError("No frames loaded")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("dt").drop_duplicates(subset=["dt"], keep="first")
    return out.set_index("dt")


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def to_ny_index(df: pd.DataFrame, data_tz: str) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(data_tz, ambiguous="infer", nonexistent="shift_forward")
    df.index = idx.tz_convert("America/New_York")
    return df


def dow_allowed(ts: pd.Timestamp, p: Params) -> bool:
    d = ts.dayofweek
    return (
        (d == 0 and p.trade_monday)
        or (d == 1 and p.trade_tuesday)
        or (d == 2 and p.trade_wednesday)
        or (d == 3 and p.trade_thursday)
        or (d == 4 and p.trade_friday)
    )


def in_opening_range(ts: pd.Timestamp, p: Params) -> bool:
    return ts.hour == p.or_start_hour and p.or_start_minute <= ts.minute <= p.or_end_minute


def session_close_bar(ts: pd.Timestamp, p: Params) -> bool:
    return ts.hour == p.session_end_hour and ts.minute == p.session_end_minute


def distance_ok(close_px: float, direction: str, or_high: float, or_low: float, p: Params) -> bool:
    if not p.use_breakout_distance_filter:
        return True
    or_size = or_high - or_low
    if direction == "long":
        mn = or_high + or_size * p.min_breakout_multiplier
        mx = or_high + or_size * p.max_breakout_multiplier
        return mn <= close_px <= mx
    mn = or_low - or_size * p.min_breakout_multiplier
    mx = or_low - or_size * p.max_breakout_multiplier
    return close_px <= mn and close_px >= mx


def wick_ok_row(o: float, h: float, l: float, c: float, direction: str, p: Params) -> bool:
    if not p.use_wick_filter:
        return True
    body = abs(c - o)
    if body == 0:
        return False
    if direction == "long":
        top_wick = h - max(o, c)
        return (top_wick / body) * 100 <= p.max_wick_percent
    bottom_wick = min(o, c) - l
    return (bottom_wick / body) * 100 <= p.max_wick_percent


def stop_price(
    direction: Direction,
    p: Params,
    or_high: float,
    or_low: float,
    bo_high: float,
    bo_low: float,
) -> float:
    tick = p.tick_size * p.sl_points
    if p.sl_type == "Breakout Candle":
        return bo_low - tick if direction == "long" else bo_high + tick
    return or_low - tick if direction == "long" else or_high + tick


def limit_entry_price(direction: Direction, bo_high: float, bo_low: float, p: Params) -> float:
    body = bo_high - bo_low
    retr = body * (p.retracement_percent / 100.0)
    return bo_high - retr if direction == "long" else bo_low + retr


@dataclass
class PendingMarket:
    direction: Direction
    signal_ts: pd.Timestamp
    bo_high: float
    bo_low: float
    or_high: float
    or_low: float
    is_second: bool
    entry_id: str


@dataclass
class PendingLimit:
    direction: Direction
    limit_price: float
    bo_high: float
    bo_low: float
    or_high: float
    or_low: float
    is_second: bool
    entry_id: str


@dataclass
class OpenPosition:
    direction: Direction
    entry_price: float
    sl: float
    tp: float
    initial_risk: float
    entry_bar_idx: int
    entry_time: pd.Timestamp
    is_second: bool
    trailing_stop: Optional[float] = None
    trailing_started: bool = False


def run_backtest_m1(df_ny: pd.DataFrame, p: Params) -> pd.DataFrame:
    df_ny = df_ny.copy()
    df_ny["atr"] = wilder_atr(df_ny["high"], df_ny["low"], df_ny["close"], p.atr_length)

    # Keep only NY session window once (~7h/day vs 24h) — massive less work in groupby + loop
    df_sess = df_ny.between_time(dt_time(9, 25), dt_time(16, 30))
    day_key = df_sess.index.normalize()

    trades: list[dict[str, Any]] = []
    for _day_stamp, day_active in df_sess.groupby(day_key, sort=False):
        if day_active.empty:
            continue
        sample_ts = day_active.index[0]
        if not dow_allowed(sample_ts, p):
            continue

        or_high = np.nan
        or_low = np.nan
        breakout_occurred = False
        bo_high = bo_low = np.nan
        breakout_price = np.nan
        breakout_direction: Optional[str] = None

        first_trade_sl_hit = False
        first_trade_direction: Optional[str] = None
        second_chance_available = False
        second_trade_taken = False
        daily_trades_complete = False

        entry_placed = False
        second_entry_placed = False
        pending_m: Optional[PendingMarket] = None
        pending_l: Optional[PendingLimit] = None
        pos: Optional[OpenPosition] = None

        for i in range(len(day_active)):
            ts = day_active.index[i]
            row = day_active.iloc[i]
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            atr_v = float(row["atr"]) if not np.isnan(row["atr"]) else np.nan

            if p.force_session_close and session_close_bar(ts, p):
                pending_m = pending_l = None
                if pos is not None:
                    _close_position(pos, o, "SessionEnd", ts, trades, p, sample_ts.normalize())
                    had_first = entry_placed and not second_entry_placed
                    pos = None
                    if had_first and not first_trade_sl_hit:
                        if p.enable_second_chance and not second_trade_taken:
                            first_trade_sl_hit = True
                            first_trade_direction = breakout_direction
                            second_chance_available = True
                        else:
                            daily_trades_complete = True

            if pos is None and pending_m is not None and ts > pending_m.signal_ts:
                direction = pending_m.direction
                entry_px = o
                sl = stop_price(
                    direction, p, pending_m.or_high, pending_m.or_low,
                    pending_m.bo_high, pending_m.bo_low,
                )
                if direction == "long":
                    risk = entry_px - sl
                    if risk <= 0:
                        pending_m = None
                    else:
                        tp = entry_px + risk * p.risk_multiplier
                        pos = OpenPosition(
                            direction=direction,
                            entry_price=entry_px,
                            sl=sl,
                            tp=tp,
                            initial_risk=abs(entry_px - sl),
                            entry_bar_idx=i,
                            entry_time=ts,
                            is_second=pending_m.is_second,
                            trailing_stop=sl if p.use_trailing_sl else None,
                            trailing_started=False,
                        )
                        pending_m = None
                else:
                    risk = sl - entry_px
                    if risk <= 0:
                        pending_m = None
                    else:
                        tp = entry_px - risk * p.risk_multiplier
                        pos = OpenPosition(
                            direction=direction,
                            entry_price=entry_px,
                            sl=sl,
                            tp=tp,
                            initial_risk=abs(entry_px - sl),
                            entry_bar_idx=i,
                            entry_time=ts,
                            is_second=pending_m.is_second,
                            trailing_stop=sl if p.use_trailing_sl else None,
                            trailing_started=False,
                        )
                        pending_m = None

            if pos is None and pending_l is not None:
                lim = pending_l.limit_price
                filled = False
                fill_px = lim
                if pending_l.direction == "long" and l <= lim <= h:
                    filled = True
                elif pending_l.direction == "short" and l <= lim <= h:
                    filled = True
                if filled:
                    direction = pending_l.direction
                    sl = stop_price(
                        direction, p, pending_l.or_high, pending_l.or_low,
                        pending_l.bo_high, pending_l.bo_low,
                    )
                    entry_px = fill_px
                    if direction == "long":
                        risk = entry_px - sl
                        if risk <= 0:
                            pending_l = None
                        else:
                            tp = entry_px + risk * p.risk_multiplier
                            pos = OpenPosition(
                                direction=direction,
                                entry_price=entry_px,
                                sl=sl,
                                tp=tp,
                                initial_risk=abs(entry_px - sl),
                                entry_bar_idx=i,
                                entry_time=ts,
                                is_second=pending_l.is_second,
                                trailing_stop=sl if p.use_trailing_sl else None,
                                trailing_started=False,
                            )
                            pending_l = None
                    else:
                        risk = sl - entry_px
                        if risk <= 0:
                            pending_l = None
                        else:
                            tp = entry_px - risk * p.risk_multiplier
                            pos = OpenPosition(
                                direction=direction,
                                entry_price=entry_px,
                                sl=sl,
                                tp=tp,
                                initial_risk=abs(entry_px - sl),
                                entry_bar_idx=i,
                                entry_time=ts,
                                is_second=pending_l.is_second,
                                trailing_stop=sl if p.use_trailing_sl else None,
                                trailing_started=False,
                            )
                            pending_l = None

            if pos is not None:
                active_sl = pos.sl
                if p.use_trailing_sl and pos.trailing_stop is not None:
                    active_sl = pos.trailing_stop

                if p.use_trailing_sl and not np.isnan(atr_v):
                    if pos.direction == "long":
                        profit_r = (c - pos.entry_price) / pos.initial_risk
                        if not pos.trailing_started and profit_r >= p.profit_r_multiplier:
                            pos.trailing_started = True
                            pos.trailing_stop = max(pos.trailing_stop or pos.sl, c - atr_v * p.atr_multiplier)
                        if pos.trailing_started:
                            pot = c - atr_v * p.atr_multiplier
                            if pot > (pos.trailing_stop or pos.sl) and pot > pos.sl:
                                pos.trailing_stop = pot
                                active_sl = pos.trailing_stop
                    else:
                        profit_r = (pos.entry_price - c) / pos.initial_risk
                        if not pos.trailing_started and profit_r >= p.profit_r_multiplier:
                            pos.trailing_started = True
                            pos.trailing_stop = min(pos.trailing_stop or pos.sl, c + atr_v * p.atr_multiplier)
                        if pos.trailing_started:
                            pot = c + atr_v * p.atr_multiplier
                            if pot < (pos.trailing_stop or pos.sl) and pot < pos.sl:
                                pos.trailing_stop = pot
                                active_sl = pos.trailing_stop

                exit_px, reason = _check_exit_intrabar(pos.direction, o, h, l, active_sl, pos.tp)
                if exit_px is not None:
                    _close_position(pos, exit_px, reason, ts, trades, p, sample_ts.normalize())
                    had_first = entry_placed and not second_entry_placed
                    pos = None
                    if had_first and not first_trade_sl_hit:
                        if p.enable_second_chance and not second_trade_taken:
                            first_trade_sl_hit = True
                            first_trade_direction = breakout_direction
                            second_chance_available = True
                        else:
                            daily_trades_complete = True

            if daily_trades_complete and pos is None and pending_m is None and pending_l is None:
                break

            if in_opening_range(ts, p):
                if np.isnan(or_high) or np.isnan(or_low):
                    or_high, or_low = h, l
                else:
                    or_high = max(or_high, h)
                    or_low = min(or_low, l)
                continue

            if np.isnan(or_high) or np.isnan(or_low):
                continue
            if not dow_allowed(ts, p):
                continue

            if not breakout_occurred and not daily_trades_complete and not in_opening_range(ts, p):
                candle_body = abs(c - o)
                top_wick_pct = ((h - max(o, c)) / candle_body * 100) if candle_body > 0 else 0.0
                bot_wick_pct = ((min(o, c) - l) / candle_body * 100) if candle_body > 0 else 0.0

                if c > or_high:
                    breakout_occurred = True
                    bo_high, bo_low = h, l
                    breakout_price = c
                    breakout_direction = "long"
                    wfv = not p.use_wick_filter or top_wick_pct <= p.max_wick_percent
                    dv = distance_ok(c, "long", or_high, or_low, p)
                    if not (wfv and dv):
                        breakout_direction = "invalid"
                elif c < or_low:
                    breakout_occurred = True
                    bo_high, bo_low = h, l
                    breakout_price = c
                    breakout_direction = "short"
                    wfv = not p.use_wick_filter or bot_wick_pct <= p.max_wick_percent
                    dv = distance_ok(c, "short", or_high, or_low, p)
                    if not (wfv and dv):
                        breakout_direction = "invalid"

            if (
                breakout_occurred
                and breakout_direction == "invalid"
                and p.enable_second_chance
                and not first_trade_sl_hit
            ):
                first_trade_sl_hit = True
                first_trade_direction = "long" if breakout_price > or_high else "short"
                second_chance_available = True

            if (
                second_chance_available
                and first_trade_direction is not None
                and not second_trade_taken
                and not daily_trades_complete
                and not in_opening_range(ts, p)
                and dow_allowed(ts, p)
            ):
                candle_body = abs(c - o)
                top_wick_pct = ((h - max(o, c)) / candle_body * 100) if candle_body > 0 else 0.0
                bot_wick_pct = ((min(o, c) - l) / candle_body * 100) if candle_body > 0 else 0.0

                triggered = False
                new_dir: Optional[Direction] = None
                if first_trade_direction == "long" and c < or_low:
                    triggered = True
                    new_dir = "short"
                    wfv = not p.use_wick_filter or bot_wick_pct <= p.max_wick_percent
                    dv = distance_ok(c, "short", or_high, or_low, p)
                elif first_trade_direction == "short" and c > or_high:
                    triggered = True
                    new_dir = "long"
                    wfv = not p.use_wick_filter or top_wick_pct <= p.max_wick_percent
                    dv = distance_ok(c, "long", or_high, or_low, p)
                else:
                    wfv = dv = True

                if triggered:
                    second_trade_taken = True
                    second_chance_available = False
                    bo_high, bo_low = h, l
                    breakout_price = c
                    if not (wfv and dv):
                        breakout_direction = "invalid"
                    else:
                        breakout_direction = new_dir

            want_entry = (
                not daily_trades_complete
                and dow_allowed(ts, p)
                and breakout_direction not in (None, "invalid")
                and pos is None
                and pending_m is None
                and pending_l is None
            )
            first_slot = breakout_occurred and not entry_placed and not np.isnan(bo_high)
            second_slot = second_trade_taken and not second_entry_placed and not np.isnan(bo_high)
            if want_entry and (first_slot or second_slot):
                is_second = second_slot
                d = breakout_direction
                assert d in ("long", "short")
                entry_id = f"{'2nd ' if is_second else ''}{d}"
                if p.entry_type == "Retracement":
                    lim_px = limit_entry_price(d, bo_high, bo_low, p)
                    pending_l = PendingLimit(
                        direction=d,
                        limit_price=lim_px,
                        bo_high=bo_high,
                        bo_low=bo_low,
                        or_high=or_high,
                        or_low=or_low,
                        is_second=is_second,
                        entry_id=entry_id,
                    )
                else:
                    pending_m = PendingMarket(
                        direction=d,
                        signal_ts=ts,
                        bo_high=bo_high,
                        bo_low=bo_low,
                        or_high=or_high,
                        or_low=or_low,
                        is_second=is_second,
                        entry_id=entry_id,
                    )
                if is_second:
                    second_entry_placed = True
                    daily_trades_complete = True
                else:
                    entry_placed = True

        pending_m = pending_l = None
        if pos is not None:
            last_ts = day_active.index[-1]
            last = day_active.iloc[-1]
            _close_position(
                pos, float(last["close"]), "EOD", last_ts, trades, p, sample_ts.normalize()
            )
            pos = None

    return pd.DataFrame(trades)


def _check_exit_intrabar(
    direction: Direction,
    o: float,
    h: float,
    l: float,
    sl: float,
    tp: float,
) -> tuple[Optional[float], str]:
    if direction == "long":
        hit_sl = l <= sl
        hit_tp = h >= tp
        if hit_sl and hit_tp:
            return sl, "SL"
        if hit_sl:
            return sl, "SL"
        if hit_tp:
            return tp, "TP"
    else:
        hit_sl = h >= sl
        hit_tp = l <= tp
        if hit_sl and hit_tp:
            return sl, "SL"
        if hit_sl:
            return sl, "SL"
        if hit_tp:
            return tp, "TP"
    return None, ""


def _close_position(
    pos: OpenPosition,
    exit_px: float,
    reason: str,
    exit_ts: pd.Timestamp,
    trades: list,
    p: Params,
    day: pd.Timestamp,
) -> None:
    if pos.direction == "long":
        pts = exit_px - pos.entry_price
    else:
        pts = pos.entry_price - exit_px
    usd = pts * p.contracts * p.dollar_per_point
    trades.append(
        {
            "date": day.date(),
            "direction": pos.direction,
            "second_chance": pos.is_second,
            "entry_time": pos.entry_time,
            "exit_time": exit_ts,
            "entry": round(pos.entry_price, 5),
            "exit": round(exit_px, 5),
            "sl": round(pos.sl, 5),
            "tp": round(pos.tp, 5),
            "reason": reason,
            "pnl_usd": round(usd, 2),
            "win": usd > 0,
        }
    )


def generate_report(trades_df: pd.DataFrame, p: Params) -> dict[str, Any]:
    if trades_df.empty:
        print("No trades found.")
        return {}
    t = trades_df.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values("date")
    t["month"] = t["date"].dt.to_period("M")
    t["cum_pnl"] = t["pnl_usd"].cumsum()
    t["equity_after"] = p.initial_capital + t["cum_pnl"]
    peak = t["equity_after"].cummax()
    dd_pct = ((t["equity_after"] - peak) / peak.replace(0, np.nan)) * 100
    max_dd = float(dd_pct.min()) if len(dd_pct) else 0.0
    gross_win = t.loc[t["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = abs(t.loc[t["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    n = len(t)
    wr = t["win"].sum() / n * 100
    period_start = t["date"].min().strftime("%b %Y")
    period_end = t["date"].max().strftime("%b %Y")
    final_eq = p.initial_capital + t["pnl_usd"].sum()

    monthly_rows: list[dict[str, Any]] = []
    for month, grp in t.groupby("month", sort=True):
        grp = grp.sort_values("date")
        net = float(grp["pnl_usd"].sum())
        trades_ct = int(len(grp))
        win_amt = float(grp.loc[grp["pnl_usd"] > 0, "pnl_usd"].sum())
        loss_amt = float(grp.loc[grp["pnl_usd"] < 0, "pnl_usd"].sum())
        loss_amt_abs = abs(loss_amt)
        eq_before = float(grp.iloc[0]["equity_after"] - grp.iloc[0]["pnl_usd"])
        ret_pct = (net / eq_before * 100.0) if eq_before != 0 else 0.0
        monthly_rows.append(
            {
                "month": str(month),
                "return_pct": round(ret_pct, 2),
                "trades": trades_ct,
                "total_win_usd": round(win_amt, 2),
                "total_loss_usd": round(loss_amt_abs, 2),
                "net_pnl_usd": round(net, 2),
                "equity_end_usd": round(float(grp.iloc[-1]["equity_after"]), 2),
            }
        )
    monthly = pd.DataFrame(monthly_rows)

    hdr = (
        f"{'Month':<10} {'Return %':>10} {'Trades':>8} {'Total win $':>14} {'Total loss $':>14} {'Net $':>12}"
    )
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr))
    print("  MONTHLY RESULTS (return % vs equity at month start)")
    print("=" * len(hdr))
    print(hdr)
    print(sep)
    for _, r in monthly.iterrows():
        print(
            f"{r['month']:<10} {r['return_pct']:>10.2f} {int(r['trades']):>8} "
            f"{r['total_win_usd']:>14,.2f} {r['total_loss_usd']:>14,.2f} {r['net_pnl_usd']:>12,.2f}"
        )
    print(sep)
    print(
        f"{'ALL':<10} {((final_eq - p.initial_capital) / p.initial_capital * 100):>10.2f} {n:>8} "
        f"{gross_win:>14,.2f} {gross_loss:>14,.2f} {(final_eq - p.initial_capital):>12,.2f}"
    )
    print("=" * len(hdr))
    print("\n  Summary")
    print(f"  Data TZ → NY     : {p.data_tz!r} → America/New_York")
    print(f"  Period           : {period_start} – {period_end}")
    print(f"  Total trades     : {n:,}")
    print(f"  Win rate         : {wr:.1f}%")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print(f"  Initial capital  : ${p.initial_capital:,.0f}")
    print(f"  Final equity     : ${final_eq:,.2f}")
    print(f"  Total return     : {((final_eq - p.initial_capital) / p.initial_capital * 100):.2f}%")
    print()

    t.to_csv(os.path.join(RESULTS_DIR, "trade_log.csv"), index=False)
    monthly.to_csv(os.path.join(RESULTS_DIR, "monthly_summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t["date"], t["equity_after"], color="#00c896", linewidth=1.5)
    ax.axhline(p.initial_capital, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Dr Praise Gold Bank — Equity ({period_start} – {period_end})\n"
        f"WR {wr:.1f}%  |  PF {pf:.2f}  |  Max DD {max_dd:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/trade_log.csv, monthly_summary.csv, equity_curve.png")
    return {"trades_df": t, "monthly": monthly, "max_dd": max_dd, "pf": pf, "wr": wr}


if __name__ == "__main__":
    p = Params()
    print("\n[1/3] Loading M1 CSVs (recursive)...")
    m1 = load_all_m1_csv(DATA_ROOT)
    print(f"  Total M1 bars: {len(m1):,}")
    print("\n[2/3] Timezone + ATR prep...")
    m1_ny = to_ny_index(m1, p.data_tz)
    max_days = int(os.environ.get("BACKTEST_MAX_DAYS", "0"))
    if max_days > 0:
        uniq = m1_ny.index.normalize().unique().sort_values()
        keep = uniq[:max_days]
        m1_ny = m1_ny[m1_ny.index.normalize().isin(keep)]
        print(f"  BACKTEST_MAX_DAYS={max_days} → {len(m1_ny):,} bars")
    print(f"  Range (NY): {m1_ny.index.min()} — {m1_ny.index.max()}")
    print("\n[3/3] Running backtest...")
    trades_df = run_backtest_m1(m1_ny, p)
    print(f"  Trades: {len(trades_df):,}")
    generate_report(trades_df, p)
