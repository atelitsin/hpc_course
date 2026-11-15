#!/usr/bin/env python3
"""Plot raw task 5 measurements and calculate median speedups."""

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as error:
    print(f"Install pandas and matplotlib: {error}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} benchmark.csv", file=sys.stderr)
        raise SystemExit(2)

    path = Path(sys.argv[1])
    data = pd.read_csv(path)
    if {"median_seconds", "sequential_seconds", "speedup"}.issubset(data.columns):
        summary = data.copy()
    else:
        sequential = (
            data[data.schedule == "sequential"]
            .groupby(["format", "n"])["seconds"]
            .median()
            .rename("sequential_seconds")
        )
        parallel = data[data.schedule != "sequential"].copy()
        parallel["median_seconds"] = parallel.groupby(
            ["format", "n", "threads", "schedule", "chunk"]
        )["seconds"].transform("median")
        parallel = parallel.join(sequential, on=["format", "n"])
        parallel["speedup"] = parallel.sequential_seconds / parallel.median_seconds
        summary = parallel.drop_duplicates(
            ["format", "n", "threads", "schedule", "chunk"]
        )
    summary.to_csv(path.with_name("summary.csv"), index=False)

    outputs = [path.with_name("summary.csv")]

    for format_name, format_data in sorted(summary.groupby("format")):
        sizes = sorted(format_data.n.unique())
        schedules = sorted(format_data.schedule.unique())

        fig, axes = plt.subplots(
            len(sizes),
            len(schedules),
            figsize=(5 * len(schedules), 3.8 * len(sizes)),
            squeeze=False,
        )
        for row, n in enumerate(sizes):
            for column, schedule in enumerate(schedules):
                axis = axes[row, column]
                schedule_data = format_data[
                    (format_data.n == n) & (format_data.schedule == schedule)
                ]
                for chunk, group in sorted(schedule_data.groupby("chunk")):
                    group = group.sort_values("threads")
                    axis.plot(
                        group.threads,
                        group.speedup,
                        marker="o",
                        label=f"chunk={chunk}",
                    )
                max_threads = int(schedule_data.threads.max())
                axis.plot(
                    [1, max_threads],
                    [1, max_threads],
                    "--",
                    color="gray",
                    label="ideal",
                )
                axis.set_title(f"n={n}, {schedule}")
                axis.set_xlabel("threads")
                axis.set_ylabel("median speedup")
                axis.set_xticks(sorted(schedule_data.threads.unique()))
                axis.grid(alpha=0.3)
                axis.legend(fontsize=8)
        fig.suptitle(f"{format_name}: speedup for all matrix sizes")
        fig.tight_layout()
        output = path.with_name(f"speedup_{format_name}.png")
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)

        fig, axes = plt.subplots(
            len(sizes),
            1,
            figsize=(10, 3.8 * len(sizes)),
            squeeze=False,
        )
        for row, n in enumerate(sizes):
            axis = axes[row, 0]
            size_data = format_data[format_data.n == n]
            for schedule, schedule_data in sorted(size_data.groupby("schedule")):
                best = (
                    schedule_data.groupby("threads")["median_seconds"]
                    .min()
                    .sort_index()
                )
                axis.plot(
                    best.index,
                    best.values,
                    marker="o",
                    label=schedule,
                )
            axis.set_title(f"n={n}")
            axis.set_xlabel("threads")
            axis.set_ylabel("median time, seconds")
            axis.set_xticks(sorted(size_data.threads.unique()))
            axis.set_yscale("log")
            axis.grid(alpha=0.3, which="both")
            axis.legend()
        fig.suptitle(f"{format_name}: best time by schedule for all sizes")
        fig.tight_layout()
        output = path.with_name(f"time_{format_name}.png")
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)

    print("Saved:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
