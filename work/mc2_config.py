from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import re

import numpy as np
import pandas as pd


DATA_DIR = Path("../data/MC2/data/")
IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

STATIC_READINGS_CSV = DATA_DIR / "StaticSensorReadings.csv"
STATIC_LOCS_CSV     = DATA_DIR / "StaticSensorLocations.csv"
MOBILE_READINGS_CSV = DATA_DIR / "MobileSensorReadings.csv"

SIM_START = pd.Timestamp("2020-04-06 00:00:00")
SIM_END   = pd.Timestamp("2020-04-10 23:59:59")


NIGHTS = [
    ("2020-04-06 00:00:00", "2020-04-06 06:00:00"),
    ("2020-04-06 18:00:00", "2020-04-07 06:00:00"),
    ("2020-04-07 18:00:00", "2020-04-08 06:00:00"),
    ("2020-04-08 18:00:00", "2020-04-09 06:00:00"),
    ("2020-04-09 18:00:00", "2020-04-10 06:00:00"),
    ("2020-04-10 18:00:00", "2020-04-10 23:59:59"),
]

STATIC_BASELINE_START = pd.Timestamp("2020-04-06 00:00:00")
STATIC_BASELINE_END   = pd.Timestamp("2020-04-08 00:00:00")

MOBILE_BASELINE_START = pd.Timestamp("2020-04-06 00:00:00")
MOBILE_BASELINE_END   = pd.Timestamp("2020-04-06 06:00:00")


def css_rgb_to_mpl(color: object) -> object:

    if not isinstance(color, str):
        return color
    m = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color.strip(), flags=re.IGNORECASE)
    if not m:
        return color
    r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (r / 255.0, g / 255.0, b / 255.0)

STATIC_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "1":  css_rgb_to_mpl("rgb(59,118,175)"),
    "4":  css_rgb_to_mpl("rgb(239,133,54)"),
    "6":  css_rgb_to_mpl("rgb(81,157,62)"),
    "9":  css_rgb_to_mpl("rgb(141,106,184)"),
    "11": css_rgb_to_mpl("rgb(197,57,50)"),
    "12": css_rgb_to_mpl("rgb(132,88,78)"),
    "13": css_rgb_to_mpl("rgb(213,126,190)"),
    "14": css_rgb_to_mpl("rgb(188,188,69)"),
    "15": css_rgb_to_mpl("rgb(88,187,204)"),
}

MOBILE_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "1":  css_rgb_to_mpl("rgb(100,186,170)"), "2":  css_rgb_to_mpl("rgb(36,90,98)"),
    "3":  css_rgb_to_mpl("rgb(65,165,238)"),  "4":  css_rgb_to_mpl("rgb(85,94,208)"),
    "5":  css_rgb_to_mpl("rgb(198,103,243)"), "6":  css_rgb_to_mpl("rgb(118,7,150)"),
    "7":  css_rgb_to_mpl("rgb(233,120,177)"), "8":  css_rgb_to_mpl("rgb(104,55,79)"),
    "9":  css_rgb_to_mpl("rgb(244,38,151)"),  "10": css_rgb_to_mpl("rgb(140,2,80)"),
    "11": css_rgb_to_mpl("rgb(77,194,84)"),   "12": css_rgb_to_mpl("rgb(5,110,18)"),
    "13": css_rgb_to_mpl("rgb(141,168,62)"),  "14": css_rgb_to_mpl("rgb(104,60,0)"),
    "15": css_rgb_to_mpl("rgb(247,147,2)"),   "16": css_rgb_to_mpl("rgb(209,31,11)"),
    "17": css_rgb_to_mpl("rgb(218,157,136)"), "18": css_rgb_to_mpl("rgb(63,76,8)"),
    "19": css_rgb_to_mpl("rgb(227,19,238)"),  "20": css_rgb_to_mpl("rgb(39,15,226)"),
    "21": css_rgb_to_mpl("rgb(100,186,170)"), "22": css_rgb_to_mpl("rgb(36,90,98)"),
    "23": css_rgb_to_mpl("rgb(65,165,238)"),  "24": css_rgb_to_mpl("rgb(85,94,208)"),
    "25": css_rgb_to_mpl("rgb(198,103,243)"), "26": css_rgb_to_mpl("rgb(118,7,150)"),
    "27": css_rgb_to_mpl("rgb(233,120,177)"), "28": css_rgb_to_mpl("rgb(104,55,79)"),
    "29": css_rgb_to_mpl("rgb(244,38,151)"),  "30": css_rgb_to_mpl("rgb(140,2,80)"),
    "31": css_rgb_to_mpl("rgb(77,194,84)"),   "32": css_rgb_to_mpl("rgb(5,110,18)"),
    "33": css_rgb_to_mpl("rgb(141,168,62)"),  "34": css_rgb_to_mpl("rgb(104,60,0)"),
    "35": css_rgb_to_mpl("rgb(247,147,2)"),   "36": css_rgb_to_mpl("rgb(209,31,11)"),
    "37": css_rgb_to_mpl("rgb(218,157,136)"), "38": css_rgb_to_mpl("rgb(63,76,8)"),
    "39": css_rgb_to_mpl("rgb(227,19,238)"),  "40": css_rgb_to_mpl("rgb(39,15,226)"),
    "41": css_rgb_to_mpl("rgb(100,186,170)"), "42": css_rgb_to_mpl("rgb(36,90,98)"),
    "43": css_rgb_to_mpl("rgb(65,165,238)"),  "44": css_rgb_to_mpl("rgb(85,94,208)"),
    "45": css_rgb_to_mpl("rgb(198,103,243)"), "46": css_rgb_to_mpl("rgb(118,7,150)"),
    "47": css_rgb_to_mpl("rgb(233,120,177)"), "48": css_rgb_to_mpl("rgb(104,55,79)"),
    "49": css_rgb_to_mpl("rgb(244,38,151)"),  "50": css_rgb_to_mpl("rgb(140,2,80)"),
}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"timestamp", "time", "datetime"}:
            rename[c] = "Timestamp"
        elif cl in {"sensor-id", "sensorid", "sensor_id", "sensor"}:
            rename[c] = "Sensor-id"
        elif cl in {"value", "cpm", "reading", "radiation"}:
            rename[c] = "Value"
        elif cl in {"lat", "latitude"}:
            rename[c] = "Lat"
        elif cl in {"long", "lon", "longitude"}:
            rename[c] = "Long"
        elif cl in {"user-id", "userid", "user_id"}:
            rename[c] = "User-id"
    return df.rename(columns=rename)

# цои
# Визуалзиация

def load_static_readings(path: Path = STATIC_READINGS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Sensor-id", "Value"])
    df = df.sort_values("Timestamp")
    df = df[(df["Timestamp"] >= SIM_START) & (df["Timestamp"] <= SIM_END)]
    df["Sensor-id"] = df["Sensor-id"].astype(int)
    return df


def load_mobile_readings(path: Path = MOBILE_READINGS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Sensor-id", "Value", "Lat", "Long"])
    df = df.sort_values("Timestamp")
    df = df[(df["Timestamp"] >= SIM_START) & (df["Timestamp"] <= SIM_END)]
    df["Sensor-id"] = df["Sensor-id"].astype(int)
    return df



def load_static_locations(path: Path = STATIC_LOCS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    if "Sensor-id" in df.columns:
        df["Sensor-id"] = df["Sensor-id"].astype(int)
    return df

def global_baseline_mean(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    cut = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
    m = float(cut["Value"].mean())
    if np.isnan(m):
        m = float(df["Value"].mean())
    return m


def cusum_pivot(df: pd.DataFrame, resample_rule: str, baseline: float) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index="Timestamp", columns="Sensor-id", values="Value", aggfunc="mean")
          .sort_index()
          .resample(resample_rule).mean()
    )
    hi = pivot.quantile(0.999)
    pivot = pivot.clip(upper=hi, axis=1)
    return (pivot - baseline).fillna(0).cumsum()


def hourly_mean_ci(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Hour"] = tmp["Timestamp"].dt.floor("h")
    g = tmp.groupby(["Hour", "Sensor-id"])["Value"]

    mean = g.mean().rename("mean").reset_index()

    def ci95(s: pd.Series) -> float:
        s = s.dropna()
        n = len(s)
        if n < 2:
            return np.nan
        return float(1.96 * s.std(ddof=1) / np.sqrt(n))

    ci = g.apply(ci95).rename("ci95").reset_index()
    return mean.merge(ci, on=["Hour", "Sensor-id"], how="left")


def despike(df: pd.DataFrame, max_cpm: float = 100.0) -> pd.DataFrame:
    return df[df["Value"] < max_cpm].copy()

def add_night_bands(ax, alpha: float = 0.10, color: str = "#669") -> None:
    for s, e in NIGHTS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=alpha, color=color, linewidth=0)