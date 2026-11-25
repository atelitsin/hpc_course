"""Построить графики времени и ускорения по CSV бенчмарка OpenMP №1."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaling_plots import plot_scaling, save


def plot_comparisons(summary, output, fixed_threads, cores):
    plot_scaling(summary, output, fixed_threads, cores, "Поиск максимума")
    parallel = summary[summary["variant"] != "seq"]
    counts = sorted(parallel["threads"].unique())
    sizes = sorted(parallel["n"].unique())
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
