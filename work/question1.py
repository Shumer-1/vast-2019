import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mc2_config as C


def _time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    def fmt(x, _=None):
        dt = mdates.num2date(x)
        if dt.hour == 0 and dt.minute == 0:
            return dt.strftime("%H:%M\n%a")
        return dt.strftime("%H:%M")

    ax.xaxis.set_major_formatter(fmt)


def plot_static_log_timeline():
    df = C.load_static_readings()
    pivot = (
        df.pivot_table(index="Timestamp", columns="Sensor-id", values="Value", aggfunc="mean")
          .sort_index()
          .resample("5s").mean()
    )

    pivot = pivot.clip(lower=0, upper=1200)

    sensor_ids = sorted([int(x) for x in pivot.columns.tolist()])
    n = len(sensor_ids)

    fig_h = max(8, 1.15 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(18, fig_h))
    if n == 1:
        axes = [axes]

    for i, (ax, sid) in enumerate(zip(axes, sensor_ids)):
        C.add_night_bands(ax, alpha=0.10)

        color = C.STATIC_COLOURS.get(str(sid))
        ax.plot(pivot.index, pivot[sid], linewidth=0.6, color=color)

        ax.set_yscale("symlog", linthresh=1, base=10)
        ax.set_ylim(0, 1200)

        ax.grid(True, which="both", alpha=0.25)

        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(0.005, 0.5, str(sid), transform=ax.transAxes,
                ha="left", va="center", fontsize=10)

        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

    _time_axis(axes[-1])
    axes[-1].set_xlabel("Time")

    fig.suptitle("Static sensor readings over time (symlog CpM)", y=0.995)

    out = C.IMAGES_DIR / "mc2StaticCpmTimeline.png"
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_static_hourly_ci():
    df = C.load_static_readings()
    agg = C.hourly_mean_ci(df)

    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()
    C.add_night_bands(ax, alpha=0.06)

    for sid, g in agg.groupby("Sensor-id"):
        g = g.sort_values("Hour")
        color = C.STATIC_COLOURS.get(str(int(sid)))
        ax.plot(g["Hour"], g["mean"], linewidth=1.5, color=color)
        ax.fill_between(
            g["Hour"].values,
            (g["mean"] - g["ci95"]).values,
            (g["mean"] + g["ci95"]).values,
            alpha=0.15,
            color=color,
        )

    ax.set_title("Static sensors: hourly mean CpM with 95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel("CpM")
    ax.grid(True, alpha=0.25)
    _time_axis(ax)

    out = C.IMAGES_DIR / "mc2StaticCpmTimelineCIs.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_static_cusum():
    df = C.load_static_readings()
    baseline = C.global_baseline_mean(df, C.STATIC_BASELINE_START, C.STATIC_BASELINE_END)
    cs = C.cusum_pivot(df, resample_rule="1min", baseline=baseline)

    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()
    C.add_night_bands(ax, alpha=0.06)

    for sid in cs.columns:
        color = C.STATIC_COLOURS.get(str(int(sid)))
        ax.plot(cs.index, cs[sid], linewidth=2, color=color)

    ax.set_title("CUSUM chart (static sensors)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative deviation from background radiation")
    ax.grid(True, alpha=0.25)
    _time_axis(ax)

    out = C.IMAGES_DIR / "mc2StaticCusum.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_mobile_cusum_all():
    df = C.load_mobile_readings()
    baseline = C.global_baseline_mean(df, C.MOBILE_BASELINE_START, C.MOBILE_BASELINE_END)

    pv = (df.pivot_table(index="Timestamp", columns="Sensor-id", values="Value", aggfunc="mean")
            .sort_index()
            .resample("1min").mean())
    cs = (pv - baseline).fillna(0).cumsum()

    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()
    C.add_night_bands(ax, alpha=0.06)

    for sid in cs.columns:
        color = C.MOBILE_COLOURS.get(str(int(sid)))
        ax.plot(cs.index, cs[sid], linewidth=0.7, alpha=0.7, color=color)

    ax.axhline(0, linewidth=1, color="black")
    ax.set_title("CUSUM chart (mobile sensors)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative deviation from background radiation")
    ax.grid(True, alpha=0.25)
    _time_axis(ax)

    out = C.IMAGES_DIR / "mc2MobileAllSensorsCusum.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_static_log_timeline()
    plot_static_hourly_ci()
    plot_static_cusum()
    plot_mobile_cusum_all()
