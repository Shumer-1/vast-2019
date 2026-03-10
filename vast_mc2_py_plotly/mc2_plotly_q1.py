from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mc2_config import (
    DATA_DIR_DEFAULT,
    OUT_DIR_DEFAULT,
    START_TS,
    END_TS,
    NIGHT_INTERVALS,
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX,
    iter_pois,
)
from mc2_paths import Mc2Paths
from mc2_io import load_mobile_readings, load_static_locations, load_static_readings
from mc2_preprocess import (
    thin_timebin_minmax,
    compute_mobile_baseline_map,
    cusum_mobile,
    thin_trajectory,
)
from mc2_colors import MOBILE_COLOURS_RGB, STATIC_COLOURS_RGB


# ---------------- Heatmap defaults (no new CLI args) ----------------

HEATMAP_THIN_MOBILE = "10min"
HEATMAP_THIN_STATIC = "10min"

HEATMAP_FRAME_FREQ = "1h"
HEATMAP_WINDOW = "2h"
HEATMAP_ENDSTATE_WINDOW = "4h"

HEATMAP_GRID_NX = 140
HEATMAP_GRID_NY = 110

HEATMAP_SIGMA = 0.008
HEATMAP_MAX_DIST = 0.018
HEATMAP_MIN_SUPPORT = 0.35

HEATMAP_STATIC_WEIGHT = 1.6
HEATMAP_MOBILE_WEIGHT = 1.0

# Мертвая зона: мелкие колебания фона не считаем аварийным превышением
HEATMAP_EXCESS_DEADBAND_STATIC = 2.5
HEATMAP_EXCESS_DEADBAND_MOBILE = 4.5

# Шкалы считаем по уже готовым полям (по кадрам), а не по исходным точкам
HEATMAP_EXCESS_ZMAX_Q_ANIM = 0.99
HEATMAP_EXCESS_ZMAX_Q_PEAK = 0.999

HEATMAP_RAW_ZMAX_Q_ANIM = 0.99
HEATMAP_RAW_ZMAX_Q_PEAK = 0.999

# Чтобы zmax не стал слишком маленьким на "спокойных" данных
HEATMAP_MIN_ZMAX_EXCESS = 6.0
HEATMAP_MIN_ZMAX_RAW = 20.0


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)

    p.add_argument("--mobile-max-rows", type=int, default=None)

    # Thinning for different pages (full 5-day timeline)
    p.add_argument("--thin-mobile-linear", default="30s", help="e.g. 10s, 30s, 1min")
    p.add_argument("--thin-mobile-log", default="30s", help="e.g. 10s, 30s, 1min (for group charts)")
    p.add_argument("--thin-mobile-cusum", default="1min", help="e.g. 30s, 1min, 2min")
    p.add_argument("--thin-static", default="1min", help="e.g. 30s, 1min, 2min")
    p.add_argument("--static-ci-freq", default="1h", help="e.g. 15min, 1h")

    p.add_argument("--despike-cap", type=float, default=100.0)
    p.add_argument("--hotspot-threshold", type=float, default=1000.0)

    return p.parse_args()


# ---------------- Helpers ----------------

def _add_night_shapes(fig: go.Figure) -> None:
    shapes = []
    for a, b in NIGHT_INTERVALS:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=a,
                x1=b,
                y0=0,
                y1=1,
                fillcolor="rgba(100,100,140,0.08)",
                line=dict(width=0),
                layer="below",
            )
        )
    fig.update_layout(shapes=shapes)


def _write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", config={"responsive": True})


def _write_dashboard(out_path: Path, items: List[Tuple[str, str]]) -> None:
    # items: (title, filename)
    links = "\n".join([f"<li><a href='{fn}' target='_blank'>{title}</a></li>" for title, fn in items])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>VAST MC2 – Plotly outputs</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 18px; }}
    .note {{ color: #444; margin: 8px 0 16px 0; max-width: 1100px; }}
    code {{ background:#f2f2f2; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>VAST Challenge 2019 – MC2 (Plotly) – outputs</h1>
  <div class="note">
    Тут есть <b>все графики</b>: static/mobile timelines, CUSUM, группы, карта траекторий и новые тепловые карты
    радиационного поля. Для “ковровых” графиков (50 сенсоров × 5 суток) в Plotly нормальный workflow такой:
    зум через range slider / drag-zoom, либо открыть нужную “группу”.
  </div>
  <ul>
    {links}
  </ul>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _ci95_by_timebin(df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    # returns: sensorId, tbin, mean, ci0, ci1
    out = df.copy()
    out["tbin"] = out["timestamp"].dt.floor(freq)
    g = out.groupby(["sensorId", "tbin"])["cpm"]
    agg = g.agg(n="count", mean="mean", std="std").reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci0"] = agg["mean"] - 1.96 * agg["se"]
    agg["ci1"] = agg["mean"] + 1.96 * agg["se"]
    return agg


def _build_facets_timeseries(
    df: pd.DataFrame,
    sensor_ids: List[int],
    y_col: str,
    title: str,
    colors: Dict[int, str],
    yaxis_type: str = "linear",   # "linear" or "log"
    y_range: Tuple[float, float] | None = None,
    baseline_line: float | None = None,
    cap_for_log_zero: float = 0.1,
    height_per_row: int = 80,
) -> go.Figure:
    """
    One subplot row per sensorId.
    """
    n = len(sensor_ids)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.002,
        subplot_titles=[str(sid) for sid in sensor_ids],
    )

    for r, sid in enumerate(sensor_ids, start=1):
        g = df[df["sensorId"] == sid].sort_values("timestamp")
        y = g[y_col].astype(float).to_numpy()

        if yaxis_type == "log":
            # Plotly log axis cannot show <=0; clamp.
            y = np.where(y <= 0, cap_for_log_zero, y)

        fig.add_trace(
            go.Scattergl(
                x=g["timestamp"],
                y=y,
                mode="lines",
                line=dict(width=1, color=colors.get(sid, "rgb(80,80,80)")),
                hovertemplate=f"sensor={sid}<br>%{{x}}<br>{y_col}=%{{y}}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=1,
        )

        # baseline line for each row
        if baseline_line is not None:
            fig.add_hline(
                y=baseline_line if (yaxis_type != "log" or baseline_line > 0) else cap_for_log_zero,
                line_width=1,
                line_color="black",
                opacity=0.6,
                row=r,
                col=1,
            )

        # y axis config per row
        if yaxis_type == "log":
            fig.update_yaxes(type="log", row=r, col=1)
            if y_range is not None:
                # plotly log axis range is log10
                lo = max(y_range[0], cap_for_log_zero)
                hi = max(y_range[1], lo * 10)
                fig.update_yaxes(range=[np.log10(lo), np.log10(hi)], row=r, col=1)
        else:
            if y_range is not None:
                fig.update_yaxes(range=list(y_range), row=r, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(450, n * height_per_row + 120),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(range=[START_TS, END_TS], row=n, col=1)
    _add_night_shapes(fig)
    return fig


def _build_overlay_lines(
    df: pd.DataFrame,
    sensor_ids: List[int],
    x_col: str,
    y_col: str,
    title: str,
    colors: Dict[int, str],
    yaxis_title: str,
    show_rangeslider: bool = True,
) -> go.Figure:
    fig = go.Figure()
    for sid in sensor_ids:
        g = df[df["sensorId"] == sid].sort_values(x_col)
        fig.add_trace(
            go.Scattergl(
                x=g[x_col],
                y=g[y_col],
                mode="lines",
                line=dict(width=1, color=colors.get(sid, "rgb(80,80,80)")),
                opacity=0.85,
                name=str(sid),
                hovertemplate=f"sensor={sid}<br>%{{x}}<br>{y_col}=%{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(
            range=[START_TS, END_TS],
            type="date",
            rangeslider=dict(visible=show_rangeslider, thickness=0.12),
        ),
        yaxis=dict(title=yaxis_title),
        margin=dict(l=60, r=30, t=60, b=60),
        height=720,
        showlegend=True,
        legend=dict(title="sensorId", orientation="v"),
    )
    _add_night_shapes(fig)
    return fig


def _build_static_ci_overlay(ci_df: pd.DataFrame, sensor_ids: List[int], title: str) -> go.Figure:
    fig = go.Figure()
    for sid in sensor_ids:
        g = ci_df[ci_df["sensorId"] == sid].sort_values("tbin")
        color = STATIC_COLOURS_RGB.get(int(sid), "rgb(80,80,80)")

        # area between ci0 and ci1
        x = g["tbin"]
        fig.add_trace(
            go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([g["ci1"], g["ci0"][::-1]]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.30)"),
                line=dict(color=color, width=0.5),
                name=str(sid),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(range=[START_TS, END_TS], type="date", rangeslider=dict(visible=True, thickness=0.12)),
        yaxis=dict(title="CpM", range=[14, 32]),
        margin=dict(l=60, r=30, t=60, b=60),
        height=600,
        showlegend=True,
        legend=dict(title="sensorId"),
    )
    _add_night_shapes(fig)
    return fig


def _build_trajectories_hotspots(
    mobile_full: pd.DataFrame,
    mobile_thin_for_hotspots: pd.DataFrame,
    static_locations: pd.DataFrame,
    hotspot_threshold: float,
    title: str,
) -> go.Figure:
    sensor_ids = sorted(mobile_full["sensorId"].dropna().astype(int).unique().tolist())
    traj = thin_trajectory(mobile_full)
    hot = mobile_thin_for_hotspots[mobile_thin_for_hotspots["cpm"] >= hotspot_threshold].copy()

    fig = go.Figure()

    def sensor_label(sid: int) -> str:
        if "userId" in mobile_full.columns:
            vals = (
                mobile_full.loc[mobile_full["sensorId"] == sid, "userId"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if vals:
                return f"{sid} — {vals[0]}"
        return f"{sid}"

    for sid in sensor_ids:
        color = MOBILE_COLOURS_RGB.get(sid, "rgb(120,120,120)")
        label = sensor_label(sid)
        legend_group = f"sensor_{sid}"

        # ---- trajectory ----
        g_traj = traj[traj["sensorId"] == sid].sort_values("timestamp")
        if len(g_traj) > 0:
            fig.add_trace(
                go.Scattergl(
                    x=g_traj["long"],
                    y=g_traj["lat"],
                    mode="lines",
                    line=dict(width=1, color=color),
                    opacity=0.25,
                    name=label,
                    legendgroup=legend_group,
                    showlegend=True,
                    hovertemplate=(
                        f"sensor={sid}"
                        "<br>lon=%{x}"
                        "<br>lat=%{y}"
                        "<extra></extra>"
                    ),
                )
            )

        # ---- hotspots of the same sensor ----
        g_hot = hot[hot["sensorId"] == sid].sort_values("timestamp")
        if len(g_hot) > 0:
            cpm = g_hot["cpm"].astype(float).to_numpy()
            c = cpm.clip(1000.0, 50000.0)

            # как в Vega/LitVis: степенная шкала exponent=0.6
            t = (c - 1000.0) / (50000.0 - 1000.0)

            # В Vega size = площадь круга, не диаметр
            areas = 400.0 + (t ** 0.6) * (40000.0 - 400.0)

            # Plotly marker.size = диаметр в пикселях
            sizes = 2.0 * np.sqrt(areas / np.pi)

            fig.add_trace(
                go.Scattergl(
                    x=g_hot["long"],
                    y=g_hot["lat"],
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=color,
                        opacity=0.10,
                    ),
                    name=label,
                    legendgroup=legend_group,
                    showlegend=False,
                    text=g_hot["sensorId"].astype(int),
                    customdata=g_hot["cpm"],
                    hovertemplate=(
                        "sensor=%{text}"
                        "<br>cpm=%{customdata}"
                        "<br>lon=%{x}"
                        "<br>lat=%{y}"
                        "<extra></extra>"
                    ),
                )
            )

    # static sensors
    fig.add_trace(
        go.Scatter(
            x=static_locations["long"],
            y=static_locations["lat"],
            mode="markers+text",
            name="static sensors",
            marker=dict(
                size=10,
                color="rgba(255,0,0,0.0)",
                line=dict(width=2, color="red"),
            ),
            text=static_locations["id"].astype(str),
            textposition="middle right",
            hovertemplate="static %{text}<extra></extra>",
            showlegend=True,
        )
    )

    # POIs
    pois = list(iter_pois())
    fig.add_trace(
        go.Scatter(
            x=[p.lon for p in pois],
            y=[p.lat for p in pois],
            mode="text",
            text=[p.label for p in pois],
            textfont=dict(size=18),
            hoverinfo="skip",
            name="POI",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(title="Longitude"),
        yaxis=dict(title="Latitude", scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=220, t=60, b=60),
        height=850,
        legend=dict(
            title="Mobile sensors",
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            groupclick="togglegroup",
        ),
    )
    return fig


# ---------------- Heatmap helpers ----------------

def _compute_static_baseline_map(
    df_static_full: pd.DataFrame,
    baseline_cutoff: pd.Timestamp = pd.Timestamp("2020-04-08 00:00:00"),
) -> Tuple[Dict[int, float], float]:
    early = df_static_full[df_static_full["timestamp"] < baseline_cutoff].copy()
    if len(early) == 0:
        fallback = float(df_static_full["cpm"].mean())
        return {}, fallback

    per_sensor = early.groupby("sensorId")["cpm"].mean().to_dict()
    fallback = float(early["cpm"].mean())
    return {int(k): float(v) for k, v in per_sensor.items()}, fallback


def _attach_heatmap_value_column(
    df: pd.DataFrame,
    baseline_map: Dict[int, float],
    fallback_baseline: float,
    mode: str,
    obs_weight: float,
    deadband: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()

    def baseline_for_sid(sid: int) -> float:
        b = float(baseline_map.get(int(sid), np.nan))
        if not np.isfinite(b) or b == 0.0:
            return fallback_baseline
        return b

    out["baseline"] = out["sensorId"].map(baseline_for_sid).astype(float)

    if mode == "raw":
        out["value"] = out["cpm"].clip(lower=0)
    else:
        # Превышением считаем только то, что выше baseline + deadband
        out["value"] = (out["cpm"] - out["baseline"] - deadband).clip(lower=0)

    out["obs_weight"] = float(obs_weight)
    return out

def _field_positive_quantile(
    z_arrays: List[np.ndarray],
    q: float,
    fallback: float,
) -> float:
    vals_parts: List[np.ndarray] = []

    for z in z_arrays:
        a = np.asarray(z, dtype=float)
        mask = np.isfinite(a) & (a > 0)
        if mask.any():
            vals_parts.append(a[mask].ravel())

    if not vals_parts:
        return float(fallback)

    vals = np.concatenate(vals_parts)
    zmax = float(np.quantile(vals, q))

    if not np.isfinite(zmax) or zmax <= 0:
        return float(fallback)

    return max(float(fallback), zmax)

def _dynamic_positive_quantile(
    s: pd.Series,
    q: float,
    fallback: float,
) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if len(vals) == 0:
        return float(fallback)

    z = float(vals.quantile(q))
    if not np.isfinite(z) or z <= 0:
        return float(fallback)

    return max(float(fallback), z)


def _prepare_heatmap_points(
    static_full: pd.DataFrame,
    static_locations: pd.DataFrame,
    mobile_full: pd.DataFrame,
) -> pd.DataFrame:
    # Baselines from FULL data (before thinning)
    static_baseline_map, static_fallback = _compute_static_baseline_map(static_full)
    mobile_baseline_map = compute_mobile_baseline_map(mobile_full)
    mobile_early = mobile_full[mobile_full["timestamp"] < pd.Timestamp("2020-04-06 06:00:00")]
    mobile_fallback = float(mobile_early["cpm"].mean()) if len(mobile_early) > 0 else float(mobile_full["cpm"].mean())

    # Thinning for heatmaps
    static_thin = thin_timebin_minmax(static_full, freq=HEATMAP_THIN_STATIC)
    mobile_thin = thin_timebin_minmax(mobile_full, freq=HEATMAP_THIN_MOBILE)

    # Static points need coords from location file
    static_pts = static_thin.merge(
        static_locations.rename(columns={"id": "sensorId"}),
        on="sensorId",
        how="left",
    ).dropna(subset=["long", "lat"]).copy()

    # --- excess (with deadband) ---
    static_pts_excess = _attach_heatmap_value_column(
        static_pts,
        baseline_map=static_baseline_map,
        fallback_baseline=static_fallback,
        mode="excess",
        obs_weight=HEATMAP_STATIC_WEIGHT,
        deadband=HEATMAP_EXCESS_DEADBAND_STATIC,
    ).assign(source="static", mode="excess")

    mobile_pts_excess = _attach_heatmap_value_column(
        mobile_thin,
        baseline_map=mobile_baseline_map,
        fallback_baseline=mobile_fallback,
        mode="excess",
        obs_weight=HEATMAP_MOBILE_WEIGHT,
        deadband=HEATMAP_EXCESS_DEADBAND_MOBILE,
    ).assign(source="mobile", mode="excess")

    # --- raw ---
    static_pts_raw = _attach_heatmap_value_column(
        static_pts,
        baseline_map=static_baseline_map,
        fallback_baseline=static_fallback,
        mode="raw",
        obs_weight=HEATMAP_STATIC_WEIGHT,
        deadband=0.0,
    ).assign(source="static", mode="raw")

    mobile_pts_raw = _attach_heatmap_value_column(
        mobile_thin,
        baseline_map=mobile_baseline_map,
        fallback_baseline=mobile_fallback,
        mode="raw",
        obs_weight=HEATMAP_MOBILE_WEIGHT,
        deadband=0.0,
    ).assign(source="mobile", mode="raw")

    cols = ["timestamp", "sensorId", "long", "lat", "cpm", "baseline", "value", "obs_weight", "source", "mode"]

    pts = pd.concat(
        [
            static_pts_excess[cols],
            mobile_pts_excess[cols],
            static_pts_raw[cols],
            mobile_pts_raw[cols],
        ],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)

    return pts


def _gaussian_field(
    points_lon: np.ndarray,
    points_lat: np.ndarray,
    values: np.ndarray,
    point_weights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    sigma: float,
    max_dist: float,
    min_support: float,
) -> np.ndarray:
    """
    Regular-grid Gaussian-kernel interpolation.

    points_lon, points_lat, values, point_weights: 1D arrays, same length
    xs, ys: grid coordinates
    returns z with shape [len(ys), len(xs)]
    """
    if len(values) == 0:
        return np.full((len(ys), len(xs)), np.nan, dtype=float)

    X, Y = np.meshgrid(xs, ys)

    gx = X.ravel()[:, None]
    gy = Y.ravel()[:, None]
    px = points_lon[None, :]
    py = points_lat[None, :]

    d2 = (gx - px) ** 2 + (gy - py) ** 2
    nearest = np.sqrt(d2.min(axis=1))

    kernel = np.exp(-d2 / (2.0 * sigma * sigma))
    kernel = kernel * point_weights[None, :]

    support = kernel.sum(axis=1)
    z = (kernel * values[None, :]).sum(axis=1) / np.clip(support, 1e-12, None)

    mask = (support < min_support) | (nearest > max_dist)
    z[mask] = np.nan

    return z.reshape(len(ys), len(xs))


def _compute_radiation_field_for_time(
    pts: pd.DataFrame,
    t: pd.Timestamp,
    window: str,
    mode: str,
    zmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(LON_MIN, LON_MAX, HEATMAP_GRID_NX)
    ys = np.linspace(LAT_MIN, LAT_MAX, HEATMAP_GRID_NY)

    window_delta = pd.Timedelta(window)
    left = t - window_delta

    g = pts[
        (pts["mode"] == mode) &
        (pts["timestamp"] > left) &
        (pts["timestamp"] <= t)
    ].copy()

    if mode == "excess":
        g = g[g["value"] > 0].copy()

    if len(g) == 0:
        z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        return xs, ys, z

    z = _gaussian_field(
        points_lon=g["long"].to_numpy(dtype=float),
        points_lat=g["lat"].to_numpy(dtype=float),
        values=g["value"].to_numpy(dtype=float),
        point_weights=g["obs_weight"].to_numpy(dtype=float),
        xs=xs,
        ys=ys,
        sigma=HEATMAP_SIGMA,
        max_dist=HEATMAP_MAX_DIST,
        min_support=HEATMAP_MIN_SUPPORT,
    )
    z = np.clip(z, 0, zmax)
    return xs, ys, z


def _add_heatmap_overlays(
    fig: go.Figure,
    static_locations: pd.DataFrame,
    mobile_full: pd.DataFrame,
) -> None:
    # 1) Все траектории как тонкий контекстный слой
    traj = thin_trajectory(mobile_full)

    xs: List[object] = []
    ys: List[object] = []

    for _, g in traj.groupby("sensorId", sort=False):
        g = g.sort_values("timestamp")
        xs.extend(g["long"].tolist())
        ys.extend(g["lat"].tolist())
        xs.append(None)
        ys.append(None)

    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=0.8, color="rgba(30,30,30,0.22)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 2) Static sensors — без legend, только как опорные точки
    fig.add_trace(
        go.Scatter(
            x=static_locations["long"],
            y=static_locations["lat"],
            mode="markers+text",
            marker=dict(
                size=9,
                color="rgba(255,255,255,0.0)",
                line=dict(width=2, color="red"),
            ),
            text=static_locations["id"].astype(str),
            textposition="middle right",
            hovertemplate="static %{text}<extra></extra>",
            showlegend=False,
        )
    )

    # 3) POI — тоже без legend
    pois = list(iter_pois())
    fig.add_trace(
        go.Scatter(
            x=[p.lon for p in pois],
            y=[p.lat for p in pois],
            mode="text",
            text=[p.label for p in pois],
            textfont=dict(size=18),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _build_animated_radiation_field(
    pts: pd.DataFrame,
    static_locations: pd.DataFrame,
    mobile_full: pd.DataFrame,
    mode: str,
    frame_freq: str,
    window: str,
    title: str,
) -> go.Figure:
    frame_times = pd.date_range(
        start=pd.Timestamp(START_TS),
        end=pd.Timestamp(END_TS),
        freq=pd.Timedelta(frame_freq),
    )

    # 1) Сначала считаем ВСЕ поля по кадрам
    frame_payload: List[Tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]] = []
    z_arrays: List[np.ndarray] = []

    for t in frame_times:
        xs, ys, z = _compute_radiation_field_for_time(
            pts=pts,
            t=t,
            window=window,
            mode=mode,
            zmax=np.inf,   # тут не режем — только считаем поле
        )
        frame_payload.append((t, xs, ys, z))
        z_arrays.append(z)

    # 2) Подбираем zmax именно по ПОЛЯМ, а не по сырым точкам
    if mode == "excess":
        zmax = _field_positive_quantile(
            z_arrays,
            q=HEATMAP_EXCESS_ZMAX_Q_ANIM,
            fallback=HEATMAP_MIN_ZMAX_EXCESS,
        )
        metric_label = "above baseline"
    else:
        zmax = _field_positive_quantile(
            z_arrays,
            q=HEATMAP_RAW_ZMAX_Q_ANIM,
            fallback=HEATMAP_MIN_ZMAX_RAW,
        )
        metric_label = "raw CpM"

    # 3) Первый кадр
    _, xs0, ys0, z0_raw = frame_payload[0]
    z0 = np.clip(z0_raw, 0, zmax)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=xs0,
            y=ys0,
            z=z0,
            colorscale="YlOrRd",
            zmin=0,
            zmax=zmax,
            opacity=0.72,
            showscale=True,
            zsmooth="best",
            colorbar=dict(
                title=metric_label,
                x=1.02,
                y=0.5,
                len=0.82,
                thickness=24,
                outlinewidth=0,
            ),
            hovertemplate=(
                "lon=%{x:.4f}<br>"
                "lat=%{y:.4f}<br>"
                f"estimated {metric_label}=%{{z:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    _add_heatmap_overlays(fig, static_locations, mobile_full)

    # 4) Frames уже рендерим с фиксированным zmax
    frames: List[go.Frame] = []
    for t, xs, ys, z_raw in frame_payload:
        z = np.clip(z_raw, 0, zmax)
        frames.append(
            go.Frame(
                name=t.strftime("%Y-%m-%d %H:%M"),
                data=[
                    go.Heatmap(
                        x=xs,
                        y=ys,
                        z=z,
                        colorscale="YlOrRd",
                        zmin=0,
                        zmax=zmax,
                        opacity=0.72,
                        showscale=True,
                        zsmooth="best",
                    )
                ],
                traces=[0],
            )
        )

    fig.frames = frames

    slider_steps = [
        {
            "label": f.name[5:],
            "method": "animate",
            "args": [
                [f.name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for f in frames
    ]

    fig.update_layout(
        title=f"{title} (frame-based zmax={zmax:.2f})",
        template="plotly_white",
        xaxis=dict(
            title="Longitude",
            range=[LON_MIN, LON_MAX],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            range=[LAT_MIN, LAT_MAX],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        height=900,
        margin=dict(l=60, r=120, t=80, b=80),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 350, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.10,
                "y": -0.04,
                "len": 0.86,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"b": 10, "t": 30},
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "steps": slider_steps,
            }
        ],
    )

    return fig


def _build_peak_radiation_field(
    pts: pd.DataFrame,
    static_locations: pd.DataFrame,
    mobile_full: pd.DataFrame,
    mode: str,
    frame_freq: str,
    window: str,
    title: str,
) -> go.Figure:
    frame_times = pd.date_range(
        start=pd.Timestamp(START_TS),
        end=pd.Timestamp(END_TS),
        freq=pd.Timedelta(frame_freq),
    )

    xs = np.linspace(LON_MIN, LON_MAX, HEATMAP_GRID_NX)
    ys = np.linspace(LAT_MIN, LAT_MAX, HEATMAP_GRID_NY)

    z_arrays: List[np.ndarray] = []
    for t in frame_times:
        _, _, z = _compute_radiation_field_for_time(
            pts=pts,
            t=t,
            window=window,
            mode=mode,
            zmax=np.inf,
        )
        z_arrays.append(z)

    if z_arrays:
        stack = np.stack([np.nan_to_num(z, nan=-np.inf) for z in z_arrays], axis=0)
        z_peak = np.max(stack, axis=0)
        z_peak[np.isneginf(z_peak)] = np.nan
    else:
        z_peak = np.full((len(ys), len(xs)), np.nan, dtype=float)

    if mode == "excess":
        zmax = _field_positive_quantile(
            [z_peak],
            q=HEATMAP_EXCESS_ZMAX_Q_PEAK,
            fallback=HEATMAP_MIN_ZMAX_EXCESS,
        )
        metric_label = "peak above baseline"
    else:
        zmax = _field_positive_quantile(
            [z_peak],
            q=HEATMAP_RAW_ZMAX_Q_PEAK,
            fallback=HEATMAP_MIN_ZMAX_RAW,
        )
        metric_label = "peak raw CpM"

    z_disp = np.clip(z_peak, 0, zmax)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=z_disp,
            colorscale="YlOrRd",
            zmin=0,
            zmax=zmax,
            opacity=0.72,
            showscale=True,
            zsmooth="best",
            colorbar=dict(
                title=metric_label,
                x=1.02,
                y=0.5,
                len=0.82,
                thickness=24,
                outlinewidth=0,
            ),
            hovertemplate=(
                "lon=%{x:.4f}<br>"
                "lat=%{y:.4f}<br>"
                f"estimated {metric_label}=%{{z:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    _add_heatmap_overlays(fig, static_locations, mobile_full)

    fig.update_layout(
        title=f"{title} (frame-based zmax={zmax:.2f})",
        template="plotly_white",
        xaxis=dict(
            title="Longitude",
            range=[LON_MIN, LON_MAX],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            range=[LAT_MIN, LAT_MAX],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        height=900,
        margin=dict(l=60, r=120, t=80, b=60),
    )
    return fig


def _build_endstate_radiation_field(
    pts: pd.DataFrame,
    static_locations: pd.DataFrame,
    mobile_full: pd.DataFrame,
    mode: str,
    window: str,
    title: str,
) -> go.Figure:
    xs, ys, z = _compute_radiation_field_for_time(
        pts=pts,
        t=pd.Timestamp(END_TS),
        window=window,
        mode=mode,
        zmax=np.inf,
    )

    if mode == "excess":
        zmax = _field_positive_quantile(
            [z],
            q=HEATMAP_EXCESS_ZMAX_Q_ANIM,
            fallback=HEATMAP_MIN_ZMAX_EXCESS,
        )
        metric_label = "end-state above baseline"
    else:
        zmax = _field_positive_quantile(
            [z],
            q=HEATMAP_RAW_ZMAX_Q_ANIM,
            fallback=HEATMAP_MIN_ZMAX_RAW,
        )
        metric_label = "end-state raw CpM"

    z_disp = np.clip(z, 0, zmax)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=z_disp,
            colorscale="YlOrRd",
            zmin=0,
            zmax=zmax,
            opacity=0.72,
            showscale=True,
            zsmooth="best",
            colorbar=dict(
                title=metric_label,
                x=1.02,
                y=0.5,
                len=0.82,
                thickness=24,
                outlinewidth=0,
            ),
            hovertemplate=(
                "lon=%{x:.4f}<br>"
                "lat=%{y:.4f}<br>"
                f"estimated {metric_label}=%{{z:.2f}}"
                "<extra></extra>"
            ),
        )
    )

    _add_heatmap_overlays(fig, static_locations, mobile_full)

    fig.update_layout(
        title=f"{title} (frame-based zmax={zmax:.2f})",
        template="plotly_white",
        xaxis=dict(
            title="Longitude",
            range=[LON_MIN, LON_MAX],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            range=[LAT_MIN, LAT_MAX],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        height=900,
        margin=dict(l=60, r=120, t=80, b=60),
    )
    return fig


# ---------------- Main ----------------

def main() -> None:
    args = parse_args()

    paths = Mc2Paths(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
    )
    paths.ensure_out_dir()

    # Load full datasets
    static_full = load_static_readings(paths.static_readings_csv)
    mobile_full = load_mobile_readings(paths.mobile_csv, max_rows=args.mobile_max_rows)
    static_locations = load_static_locations(paths.static_locations_csv)

    # Sensor sets
    static_ids = [1, 4, 6, 9, 11, 12, 13, 14, 15]
    mobile_ids = list(range(1, 51))

    # Thinning
    static_thin = thin_timebin_minmax(static_full.rename(columns={"cpm": "cpm"}), freq=args.thin_static)

    mobile_linear = thin_timebin_minmax(mobile_full, freq=args.thin_mobile_linear)
    mobile_log = thin_timebin_minmax(mobile_full, freq=args.thin_mobile_log)
    mobile_cusum_base = thin_timebin_minmax(mobile_full, freq=args.thin_mobile_cusum)

    # Mobile CUSUM baseline from FULL data
    baseline_map = compute_mobile_baseline_map(mobile_full)
    mobile_cusum = cusum_mobile(mobile_cusum_base, baseline_map=baseline_map)

    # Static CI
    static_ci = _ci95_by_timebin(static_full, freq=args.static_ci_freq)

    # ---------------- Standard figures ----------------

    fig_static_cpm = _build_facets_timeseries(
        df=static_thin,
        sensor_ids=static_ids,
        y_col="cpm",
        title=f"Static sensors: CpM (log; <=0 clamped), thin={args.thin_static}",
        colors=STATIC_COLOURS_RGB,
        yaxis_type="log",
        y_range=(0.1, 1200),
        baseline_line=15.0,
        height_per_row=110,
    )

    fig_static_cusum = _build_overlay_lines(
        df=static_full.assign(
            cusum=(
                static_full["cpm"] -
                static_full[static_full["timestamp"] < "2020-04-08 00:00:00"]["cpm"].mean()
            ).groupby(static_full["sensorId"]).cumsum()
        ),
        sensor_ids=static_ids,
        x_col="timestamp",
        y_col="cusum",
        title="Static sensors: CUSUM overlay",
        colors=STATIC_COLOURS_RGB,
        yaxis_title="CUSUM",
        show_rangeslider=True,
    )

    fig_static_ci = _build_static_ci_overlay(
        ci_df=static_ci.rename(columns={"tbin": "tbin"}),
        sensor_ids=static_ids,
        title=f"Static sensors: CI bands (mean ± 1.96*SE), bin={args.static_ci_freq}",
    )

    fig_mobile_linear_facets = _build_facets_timeseries(
        df=mobile_linear[mobile_linear["cpm"] < args.despike_cap],
        sensor_ids=mobile_ids,
        y_col="cpm",
        title=f"Mobile sensors: CpM linear (despiked cpm<{args.despike_cap:g}), thin={args.thin_mobile_linear}",
        colors=MOBILE_COLOURS_RGB,
        yaxis_type="linear",
        y_range=(0, args.despike_cap),
        baseline_line=30.0,
        height_per_row=55,
    )

    fig_mobile_cusum = _build_overlay_lines(
        df=mobile_cusum,
        sensor_ids=mobile_ids,
        x_col="timestamp",
        y_col="cusum",
        title=f"Mobile sensors: CUSUM overlay (thin={args.thin_mobile_cusum}, baseline from full data)",
        colors=MOBILE_COLOURS_RGB,
        yaxis_title="CUSUM",
        show_rangeslider=True,
    )

    groups = [
        ("mc2MobileEarlyTermination", [1, 6, 23, 26, 34, 35, 47, 48, 49]),
        ("mc2MobileWorkingGaps", [5, 6, 16, 18, 28, 37, 50]),
        ("mc2MobileWedPM", [10, 42, 43, 3, 5, 16, 21, 22, 33]),
        ("mc2Mobile12", [12]),
        ("mc2MobileWilson", [29, 30, 45, 21, 25, 27, 28, 22, 24]),
    ]

    group_figs: List[Tuple[str, go.Figure]] = []
    for name, ids in groups:
        fig = _build_facets_timeseries(
            df=mobile_log,
            sensor_ids=ids,
            y_col="cpm",
            title=f"{name}: Mobile CpM (log; <=0 clamped), thin={args.thin_mobile_log}",
            colors=MOBILE_COLOURS_RGB,
            yaxis_type="log",
            y_range=(0.1, 1200),
            baseline_line=None,
            height_per_row=140 if len(ids) <= 3 else 120,
        )
        group_figs.append((name, fig))

    fig_map = _build_trajectories_hotspots(
        mobile_full=mobile_full,
        mobile_thin_for_hotspots=mobile_linear,
        static_locations=static_locations,
        hotspot_threshold=args.hotspot_threshold,
        title=f"Mobile trajectories + hotspots (threshold={args.hotspot_threshold:g}, hotspots thin={args.thin_mobile_linear})",
    )

    # ---------------- Heatmap figures ----------------

    heatmap_pts = _prepare_heatmap_points(
        static_full=static_full,
        static_locations=static_locations,
        mobile_full=mobile_full,
    )

    fig_heatmap_excess_anim = _build_animated_radiation_field(
        pts=heatmap_pts,
        static_locations=static_locations,
        mobile_full=mobile_full,
        mode="excess",
        frame_freq=HEATMAP_FRAME_FREQ,
        window=HEATMAP_WINDOW,
        title=(
            f"Estimated radiation field over time "
            f"(above baseline, frame={HEATMAP_FRAME_FREQ}, window={HEATMAP_WINDOW}; "
            f"static weight={HEATMAP_STATIC_WEIGHT:g}, mobile weight={HEATMAP_MOBILE_WEIGHT:g})"
        ),
    )

    fig_heatmap_raw_anim = _build_animated_radiation_field(
        pts=heatmap_pts,
        static_locations=static_locations,
        mobile_full=mobile_full,
        mode="raw",
        frame_freq=HEATMAP_FRAME_FREQ,
        window=HEATMAP_WINDOW,
        title=(
            f"Estimated radiation field over time "
            f"(raw CpM, frame={HEATMAP_FRAME_FREQ}, window={HEATMAP_WINDOW}; "
            f"static weight={HEATMAP_STATIC_WEIGHT:g}, mobile weight={HEATMAP_MOBILE_WEIGHT:g})"
        ),
    )

    fig_heatmap_peak_excess = _build_peak_radiation_field(
        pts=heatmap_pts,
        static_locations=static_locations,
        mobile_full=mobile_full,
        mode="excess",
        frame_freq=HEATMAP_FRAME_FREQ,
        window=HEATMAP_WINDOW,
        title=(
            f"Peak estimated radiation field across all frames "
            f"(above baseline, frame={HEATMAP_FRAME_FREQ}, window={HEATMAP_WINDOW})"
        ),
    )

    fig_heatmap_endstate_excess = _build_endstate_radiation_field(
        pts=heatmap_pts,
        static_locations=static_locations,
        mobile_full=mobile_full,
        mode="excess",
        window=HEATMAP_ENDSTATE_WINDOW,
        title=(
            f"Estimated end-state radiation field at end of period "
            f"(above baseline, last {HEATMAP_ENDSTATE_WINDOW})"
        ),
    )

    # ---------------- Write ALL HTMLs ----------------

    outputs: List[Tuple[str, str]] = []

    def save(fig: go.Figure, filename: str, title: str) -> None:
        _write_html(fig, paths.out_dir / filename)
        outputs.append((title, filename))

    save(fig_static_cpm, "plotly_mc2StaticCpmTimeline.html", "Static CpM timeline (facets)")
    save(fig_static_cusum, "plotly_mc2StaticCusum.html", "Static CUSUM overlay")
    save(fig_static_ci, "plotly_mc2StaticCpmTimelineCIs.html", "Static CI bands overlay")

    save(fig_mobile_linear_facets, "plotly_mc2MobileAllSensorsLinear.html", "Mobile CpM linear (facets, despiked)")
    save(fig_mobile_cusum, "plotly_mc2MobileAllSensorsCusum.html", "Mobile CUSUM overlay (all 50)")

    for name, fig in group_figs:
        save(fig, f"plotly_{name}.html", f"Mobile group: {name}")

    save(fig_map, "plotly_mc2MobileTrajectories.html", "Mobile trajectories + hotspots")

    save(
        fig_heatmap_excess_anim,
        "plotly_mc2RadiationFieldExcessAnimated.html",
        "Animated radiation field (above baseline; static + mobile)",
    )
    save(
        fig_heatmap_raw_anim,
        "plotly_mc2RadiationFieldRawAnimated.html",
        "Animated radiation field (raw CpM; static + mobile)",
    )
    save(
        fig_heatmap_peak_excess,
        "plotly_mc2RadiationFieldPeakExcess.html",
        "Peak radiation field over all frames (above baseline; static + mobile)",
    )
    save(
        fig_heatmap_endstate_excess,
        "plotly_mc2RadiationFieldEndStateExcess.html",
        "End-state radiation field (above baseline; static + mobile)",
    )

    # Dashboard with links to EVERYTHING
    dash = paths.out_dir / "plotly_mc2_dashboard.html"
    _write_dashboard(dash, outputs)

    print("Saved dashboard:", dash)
    for _, fn in outputs:
        print(" -", paths.out_dir / fn)


if __name__ == "__main__":
    main()