from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


def thin_timebin_minmax(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Fast, spike-preserving thinning:
    per (sensorId, floor(timestamp,freq)) keep min and max cpm.
    """
    d = df.sort_values(["sensorId", "timestamp"]).copy()
    d["_tbin"] = d["timestamp"].dt.floor(freq)

    g = d.groupby(["sensorId", "_tbin"], sort=False)["cpm"]
    idx = np.concatenate([g.idxmin().to_numpy(), g.idxmax().to_numpy()])
    idx = pd.Index(idx).dropna().unique()

    out = d.loc[idx].drop(columns=["_tbin"]).sort_values(["sensorId", "timestamp"]).reset_index(drop=True)
    return out


def thin_trajectory(df: pd.DataFrame, sid_col: str = "sensorId") -> pd.DataFrame:
    """
    Remove consecutive identical (long,lat) per sensor.
    """
    out = df.sort_values([sid_col, "timestamp"]).copy()
    changed = (out["long"].ne(out.groupby(sid_col)["long"].shift(1))) | (out["lat"].ne(out.groupby(sid_col)["lat"].shift(1)))
    first = out.groupby(sid_col).cumcount().eq(0)
    keep = changed | first
    return out.loc[keep].reset_index(drop=True)


def compute_mobile_baseline_map(
    df_mobile_full: pd.DataFrame,
    baseline_cutoff: datetime = datetime(2020, 4, 6, 6, 0, 0),
) -> Dict[int, float]:
    early = df_mobile_full[df_mobile_full["timestamp"] < pd.Timestamp(baseline_cutoff)]
    return early.groupby("sensorId")["cpm"].mean().to_dict()


def cusum_mobile(
    df_mobile: pd.DataFrame,
    baseline_map: Optional[Dict[int, float]] = None,
    baseline_cutoff: datetime = datetime(2020, 4, 6, 6, 0, 0),
    fallback_baseline: float = 25.86,
) -> pd.DataFrame:
    """
    CUSUM per sensor. Baseline should be computed from FULL dataset and passed in baseline_map.
    """
    df = df_mobile.sort_values(["sensorId", "timestamp"]).copy()

    if baseline_map is None:
        baseline_map = compute_mobile_baseline_map(df, baseline_cutoff=baseline_cutoff)

    def baseline_for(sid: int) -> float:
        b = float(baseline_map.get(sid, np.nan))
        if not np.isfinite(b) or b == 0.0:
            return fallback_baseline
        return b

    df["baseline"] = df["sensorId"].map(baseline_for).astype(float)
    df["cusum"] = (df["cpm"] - df["baseline"]).groupby(df["sensorId"]).cumsum()
    return df