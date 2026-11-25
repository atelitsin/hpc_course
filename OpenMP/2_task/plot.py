"""Графики времени и ускорения скалярного произведения."""

import argparse
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaling_plots import plot_scaling


def main():
    default_csv = Path(__file__).resolve().parent / "results" / "bench_dot.csv"
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
    if sequential.empty or not {"reduction"} <= set(variants["variant"]):
        parser.error("Нужны замеры seq и reduction.")
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
    plot_scaling(summary, output, args.threads, args.cores, "Скалярное произведение")


if __name__ == "__main__":
    main()
