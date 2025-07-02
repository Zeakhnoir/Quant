from __future__ import annotations
import numpy as np
import pandas as pd
import pathlib
import sys

# ─────────────── CONFIG ──────────────────────────────────
PATH_SPOT = pathlib.Path("trb_usdt_spot_export.csv")
PATH_PERP = pathlib.Path("trb_usdt_futures_export.csv")

THRESHOLD = 0.0007      # 0.07 %
WINDOW_MS = 5.0         # 5 ms pairing window
MS_TO_NS  = 1_000_000   # nanoseconds in one millisecond
# ─────────────────────────────────────────────────────────


# ─────────────── 1.  CSV loader ─────────────────────────
def load_csv(path: pathlib.Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"💥  File not found: {path.resolve()}")
    df = pd.read_csv(path)

    need = {"time", "bid_price", "ask_price"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"💥  {path.name} missing columns: {missing}")

    # tolerant ISO-8601 parser (mixed fractional seconds)
    df["time"] = pd.to_datetime(
        df["time"],
        format="mixed",      # pandas ≥ 1.4; else use format="ISO8601"
        utc=True
    )
    df.sort_values("time", inplace=True)
    return df


# ─────────────── 2.  jump detection ─────────────────────
def mid_price(df: pd.DataFrame) -> pd.Series:
    return (df["bid_price"] + df["ask_price"]) / 2.0


def detect_jumps(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    mp   = mid_price(df)
    rel  = mp.diff() / mp.shift(1)
    mask = rel.abs() >= thresh
    out  = df.loc[mask, "time"].to_frame("ts")
    out["direction"] = np.sign(rel[mask]).astype(int)
    return out.reset_index(drop=True)


# ─────────────── 3.  pairing logic ──────────────────────
def pair_events(a: pd.DataFrame, b: pd.DataFrame, win_ms: float):
    # strip timezone → int64 nanoseconds
    tsa = a["ts"].dt.tz_localize(None).view("int64")
    tsb = b["ts"].dt.tz_localize(None).view("int64")
    da, db = a["direction"].values, b["direction"].values

    paired_a = np.zeros(len(a), dtype=bool)
    paired_b = np.zeros(len(b), dtype=bool)
    lead_a   = np.zeros(len(a), dtype=bool)

    # pass 1: A leads B
    j = 0
    for i in range(len(a)):
        while j < len(b) and tsb[j] < tsa[i]:
            j += 1
        k = j
        while k < len(b) and tsb[k] - tsa[i] <= win_ms * MS_TO_NS:
            if not paired_b[k] and db[k] == da[i]:
                paired_a[i] = paired_b[k] = True
                lead_a[i]   = True
                break
            k += 1

    # pass 2: B leads A (for still-unpaired)
    i = 0
    for k in range(len(b)):
        while i < len(a) and tsa[i] < tsb[k]:
            i += 1
        j2 = i
        while j2 < len(a) and tsa[j2] - tsb[k] <= win_ms * MS_TO_NS:
            if not paired_a[j2] and da[j2] == db[k]:
                paired_a[j2] = paired_b[k] = True
                break
            j2 += 1

    return paired_a, paired_b, lead_a


# ─────────────── 4.  summary report ─────────────────────
def summary(spot: pd.DataFrame, perp: pd.DataFrame) -> dict:
    js = detect_jumps(spot, THRESHOLD)
    jp = detect_jumps(perp, THRESHOLD)

    ps, pp, spot_leads = pair_events(js, jp, WINDOW_MS)

    tot_s, tot_p = len(js), len(jp)
    together     = ps.sum() + pp.sum()
    noise_total  = (tot_s + tot_p) - together

    # breakdown of noise by market
    noise_spot = (~ps).sum()
    noise_perp = (~pp).sum()

    lead_s    = spot_leads.sum()
    lead_p    = pp.sum() - lead_s

    return dict(
        total_spot_jumps   = int(tot_s),
        total_perp_jumps   = int(tot_p),
        paired_events      = int(together // 2),  # number of *pairs*
        noise_events       = int(noise_total),
        noise_spot_jumps   = int(noise_spot),
        noise_perp_jumps   = int(noise_perp),
        spot_leads_perp    = int(lead_s),
        perp_leads_spot    = int(lead_p),
    )


# ─────────────── 5.  entry-point ────────────────────────
def main():
    spot_df = load_csv(PATH_SPOT)
    perp_df = load_csv(PATH_PERP)

    out = summary(spot_df, perp_df)

    print("\nLead–lag summary (±0.07 %, 5 ms)\n")
    for k, v in out.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
