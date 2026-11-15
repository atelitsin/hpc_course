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

    largest_n = summary.n.max()
    largest = summary[summary.n == largest_n]
    outputs = [path.with_name("summary.csv")]

    for format_name, format_data in summary.groupby("format"):
        largest_format = largest[largest.format == format_name]
        schedules = list(format_data.schedule.unique())

        fig, axes = plt.subplots(
            len(schedules),
            1,
            figsize=(10, 4 * len(schedules)),
            squeeze=False,
        )
        for axis, schedule in zip(axes[:, 0], schedules):
            schedule_data = largest_format[largest_format.schedule == schedule]
            for chunk, group in schedule_data.groupby("chunk"):
                group = group.sort_values("threads")
                axis.plot(
                    group.threads,
                    group.speedup,
                    marker="o",
                    label=f"chunk={chunk}",
                )
            axis.plot(
                [1, largest_format.threads.max()],
                [1, largest_format.threads.max()],
                "--",
                color="gray",
                label="ideal",
            )
            axis.set_title(f"{format_name}: {schedule}, n={largest_n}")
            axis.set_xlabel("threads")
            axis.set_ylabel("median speedup")
            axis.set_xticks(sorted(largest_format.threads.unique()))
            axis.grid(alpha=0.3)
            axis.legend()
        fig.tight_layout()
        output = path.with_name(f"speedup_{format_name}.png")
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)

        fig, axis = plt.subplots(figsize=(10, 5))
        for schedule, schedule_data in format_data.groupby("schedule"):
            best = (
                schedule_data.groupby(["n", "threads"])["median_seconds"]
                .min()
                .reset_index()
            )
            largest_schedule = best[best.n == largest_n].sort_values("threads")
            axis.plot(
                largest_schedule.threads,
                largest_schedule.median_seconds,
                marker="o",
                label=schedule,
            )
        axis.set_title(f"{format_name}: best time by schedule, n={largest_n}")
        axis.set_xlabel("threads")
        axis.set_ylabel("median time, seconds")
        axis.set_xticks(sorted(largest_format.threads.unique()))
        axis.set_yscale("log")
        axis.grid(alpha=0.3, which="both")
        axis.legend()
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
