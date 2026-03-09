from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from mc2_config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


def _read_csv_forgiving(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)


def load_static_readings(path: Path) -> pd.DataFrame:
    df = _read_csv_forgiving(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    if "timestamp" not in df.columns:
        for cand in ["Timestamp", "time", "datetime", "DateTime"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break

    if "sensorId" not in df.columns:
        for cand in ["sensor-id", "Sensor-id", "SensorId", "sensor_id", "id", "ID", "Id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "sensorId"})
                break

    if "cpm" not in df.columns:
        for cand in ["value", "Value", "reading", "Reading", "cpm "]:
            if cand in df.columns:
                df = df.rename(columns={cand: "cpm"})
                break

    required = {"timestamp", "sensorId", "cpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Static readings missing columns: {sorted(missing)}; got {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sensorId"] = pd.to_numeric(df["sensorId"], errors="coerce").astype("Int64")
    df["cpm"] = pd.to_numeric(df["cpm"], errors="coerce")

    df = df.dropna(subset=["timestamp", "sensorId", "cpm"]).copy()
    df["sensorId"] = df["sensorId"].astype(int)
    return df.sort_values(["sensorId", "timestamp"]).reset_index(drop=True)


def load_static_locations(path: Path) -> pd.DataFrame:
    df = _read_csv_forgiving(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    if "id" not in df.columns:
        for cand in ["sensorId", "Sensor-id", "sensor-id", "SensorId", "sensor_id", "ID", "Id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "id"})
                break

    if "long" not in df.columns:
        for cand in ["lng", "lon", "longitude", "Long", "Lon"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "long"})
                break
    if "lat" not in df.columns:
        for cand in ["latitude", "Lat"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "lat"})
                break

    required = {"id", "long", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Static locations missing columns: {sorted(missing)}; got {list(df.columns)}")

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["id", "long", "lat"]).copy()
    df["id"] = df["id"].astype(int)

    df = df[(df["long"].between(LON_MIN, LON_MAX)) & (df["lat"].between(LAT_MIN, LAT_MAX))].copy()
    return df.reset_index(drop=True)


def load_mobile_readings(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = _read_csv_forgiving(path, nrows=max_rows)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    ren = {}
    for c in df.columns:
        lc = c.lower().replace("_", "-")
        if lc == "timestamp":
            ren[c] = "timestamp"
        elif lc in ("sensor-id", "sensorid", "sensor"):
            ren[c] = "sensorId"
        elif lc in ("long", "lng", "lon", "longitude"):
            ren[c] = "long"
        elif lc in ("lat", "latitude"):
            ren[c] = "lat"
        elif lc in ("value", "cpm"):
            ren[c] = "cpm"
        elif lc in ("user-id", "userid", "user"):
            ren[c] = "userId"
    df = df.rename(columns=ren)

    required = {"timestamp", "sensorId", "long", "lat", "cpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mobile readings missing columns: {sorted(missing)}; got {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sensorId"] = pd.to_numeric(df["sensorId"], errors="coerce").astype("Int64")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["cpm"] = pd.to_numeric(df["cpm"], errors="coerce")

    keep_cols = ["timestamp", "sensorId", "long", "lat", "cpm"] + (["userId"] if "userId" in df.columns else [])
    df = df[keep_cols].dropna(subset=["timestamp", "sensorId", "long", "lat", "cpm"]).copy()
    df["sensorId"] = df["sensorId"].astype(int)

    df = df[df["sensorId"].between(1, 50)]
    df = df[(df["long"].between(LON_MIN, LON_MAX)) & (df["lat"].between(LAT_MIN, LAT_MAX))].copy()

    return df.sort_values(["sensorId", "timestamp"]).reset_index(drop=True)