"""Построить графики времени и ускорения по CSV бенчмарка OpenMP №1."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator


STYLES = {"seq": ("#555555", "s"), "critical": ("#ba501b", "o"),
          "reduction": ("#007398", "^")}


def draw_curve(ax, frame, x, variant, metric):
    frame = frame.sort_values(x)
    color, marker = STYLES[variant]
    if metric == "time":
        y = frame["median_seconds"] * 1e6
        ax.fill_between(frame[x], frame["q1_seconds"] * 1e6,
                        frame["q3_seconds"] * 1e6, color=color, alpha=0.17)
    else:
        y = frame["speedup"]
    ax.plot(frame[x], y.where(y > 0), color=color, marker=marker, ms=4,
            lw=1.6, label=variant, zorder=3)


def style_axes(ax, ylabel):
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", which="major", color="#d7dcdf", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(summary, output, fixed_threads, cores):
    parallel = summary[summary["variant"] != "seq"]
    seq = summary[summary["variant"] == "seq"]
    counts = sorted(parallel["threads"].unique())
    sizes = sorted(parallel["n"].unique())
    selected = sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
    fig, axes = plt.subplots(2, len(selected), squeeze=False,
                             figsize=(4.3 * len(selected), 6.8), layout="constrained")
    fig.suptitle("Поиск максимума · масштабирование по потокам", fontsize=16)
    for column, n in enumerate(selected):
        time, speed = axes[:, column]
        subset = parallel[parallel["n"] == n]
        reference = seq[seq["n"] == n].iloc[0]
        time.axhspan(reference.q1_seconds * 1e6, reference.q3_seconds * 1e6,
                     color="gray", alpha=0.12)
        time.axhline(reference.median_seconds * 1e6, color="#555555", ls="--", label="seq")
        for variant, group in subset.groupby("variant"):
            draw_curve(time, group, "threads", variant, "time")
            draw_curve(speed, group, "threads", variant, "speedup")
        speed.plot(counts, counts, "--", color="#595959", label="Идеальное S = p")
        speed.axhline(1, color="gray", ls=":")
        time.set_title(f"n = {n:,}".replace(",", " "))
        for ax, label in ((time, "Время, мкс"), (speed, "Ускорение S")):
            style_axes(ax, label)
            ax.set_xlim(counts[0] - 0.3, counts[-1] + 0.3)
            if counts[-1] > cores:
                ax.axvspan(cores, counts[-1] + 0.3, color="#87919b", alpha=0.11)
                ax.axvline(cores, color="#69737b", lw=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        speed.set_yscale("linear")
        speed.set_ylim(0, max(counts[-1], subset["speedup"].max()) * 1.05)
        if column == 0 and counts[-1] > 2:
            zoom = speed.inset_axes([0.49, 0.09, 0.47, 0.34])
            for variant, group in subset.groupby("variant"):
                draw_curve(zoom, group, "threads", variant, "speedup")
            zoom.set_yscale("log")
            zoom.tick_params(labelsize=7)
            zoom.set_title("Измерения · лог. шкала", fontsize=7)
        speed.set_xlabel("Число потоков, p")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.supxlabel(f"Полосы времени — Q1–Q3. Серый фон — больше {cores} потоков ({cores} ядер).",
                  fontsize=10)
    save(fig, output, "threads")

    fig, (time, speed) = plt.subplots(1, 2, figsize=(10.6, 4.6), layout="constrained")
    fig.suptitle(f"Поиск максимума · влияние размера при p = {fixed_threads}", fontsize=15)
    draw_curve(time, seq, "n", "seq", "time")
    for variant, group in parallel[parallel["threads"] == fixed_threads].groupby("variant"):
        draw_curve(time, group, "n", variant, "time")
        draw_curve(speed, group, "n", variant, "speedup")
    speed.axhline(fixed_threads, color="#555555", ls="--", label=f"Идеальное S = {fixed_threads}")
    speed.axhline(1, color="gray", ls=":", label="S = 1")
    for ax, label in ((time, "Время, мкс"), (speed, "Ускорение S")):
        style_axes(ax, label)
        ax.set_xscale("log")
        ax.set_xlabel("Число элементов, n")
        ax.legend(fontsize=9)
    fig.supxlabel("Точки — медианы. Полосы на графике времени — центральные 50 % замеров.",
                  fontsize=10)
    save(fig, output, "sizes")

    comparison = parallel.pivot(index=["n", "threads"], columns="variant",
                                values="median_seconds")
    comparison["critical_over_reduction"] = (
        comparison["critical"] / comparison["reduction"].where(comparison["reduction"] > 0)
    )
    comparison.to_csv(output / "reduction_comparison.csv")
    ratios = comparison["critical_over_reduction"].unstack("threads")
    logs = np.log2(ratios.where(ratios > 0))
    finite = logs.to_numpy()[np.isfinite(logs.to_numpy())]
    limit = max(0.1, abs(finite).max()) if finite.size else 1
    fig, ax = plt.subplots(figsize=(max(8, len(counts)*0.65), 4.7), layout="constrained")
    image = ax.imshow(logs, aspect="auto", cmap="PuOr", norm=TwoSlopeNorm(0, -limit, limit))
    ax.set_xticks(range(len(counts)), counts)
    ax.set_yticks(range(len(sizes)), [f"{n:,}".replace(",", " ") for n in sizes])
    for i, row in enumerate(logs.to_numpy()):
        for j, value in enumerate(row):
            if np.isfinite(value):
                ax.text(j, i, f"{2**value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.65 * limit else "#222222")
    ax.set(xlabel="Число потоков, p", ylabel="Число элементов, n",
           title="Сравнение реализаций · отношение медиан T(critical) / T(reduction)")
    fig.colorbar(image, ax=ax, format=FuncFormatter(lambda x, pos: f"{2**x:.2g}×"))
    fig.supxlabel("Выше 1 — быстрее reduction. Ниже 1 — быстрее critical.", fontsize=10)
    save(fig, output, "reduction")


def main():
    default_csv = Path(__file__).resolve().parent / "results" / "bench.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=default_csv)
    parser.add_argument("--threads", type=int, default=8,
                        help="фиксированное число потоков на графике размеров")
    parser.add_argument("--cores", type=int, default=8, help="число ядер машины замеров")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    summary = data.groupby(["variant", "n", "threads"])["seconds"].agg(
        median_seconds="median", q1_seconds=lambda sample: sample.quantile(0.25),
        q3_seconds=lambda sample: sample.quantile(0.75), samples="size",
    ).reset_index()
    sequential = summary[summary["variant"] == "seq"].set_index("n")["median_seconds"]
    variants = summary[summary["variant"] != "seq"]
    if sequential.empty or not {"critical", "reduction"} <= set(variants["variant"]):
        parser.error("Нужны замеры seq, critical и reduction.")
    if args.threads not in set(variants["threads"]):
        parser.error("В CSV нет замеров для --threads. Укажи число потоков из файла.")

    summary["sequential_seconds"] = summary["n"].map(sequential)
    summary["speedup"] = (
        summary["sequential_seconds"].where(summary["sequential_seconds"] > 0)
        / summary["median_seconds"].where(summary["median_seconds"] > 0)
    )
    missing = summary["speedup"].isna().sum()
    if missing:
        print(f"Пропущено ускорений из-за нулевой или отсутствующей медианы: {missing}")

    output = args.csv.parent
    summary.to_csv(output / "summary.csv", index=False)
    plot_comparisons(summary, output, args.threads, args.cores)


if __name__ == "__main__":
    main()
