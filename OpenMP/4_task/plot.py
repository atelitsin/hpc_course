import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "bench.csv"
stem = Path(path).stem  # bench.csv -> bench; bench_dot.csv -> bench_dot
df = pd.read_csv(path)

# база для каждого размера: медиана последовательной версии
seq = df[df.variant == "seq"].groupby("n").seconds.median().rename("seq")

par = df[df.variant != "seq"]
g = par.groupby(["variant", "n", "threads"]).seconds
res = pd.DataFrame({"med": g.median(), "tmin": g.min(), "tmax": g.max()}).reset_index()
res = res.join(seq, on="n")
res["speedup"] = res.seq / res.med
res["speedup_lo"] = res.seq / res.tmax   # худший прогон
res["speedup_hi"] = res.seq / res.tmin   # лучший прогон
res.to_csv(f"{stem}_summary.csv", index=False)

max_t = int(res.threads.max())
sizes = sorted(res.n.unique())

# для печати: ускорение при максимуме потоков
last = res[res.threads == max_t].pivot(index="n", columns="variant", values="speedup")
print(f"Ускорение при {max_t} потоках:")
print(last.round(2).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# слева: ускорение от числа потоков, по линии на каждый размер (reduction)
red = res[res.variant == "reduction"]
for n in sizes:
    d = red[red.n == n]
    ax1.plot(d.threads, d.speedup, "o-", label=f"n = {n:,}".replace(",", " "))
ax1.plot(range(1, max_t + 1), range(1, max_t + 1), "--", color="gray", label="идеальное")
ax1.set_xlabel("число потоков")
ax1.set_ylabel("ускорение относительно последовательной версии")
ax1.set_title("reduction: ускорение по потокам")
ax1.legend(fontsize=8)
ax1.grid(True)

# справа: ускорение при максимуме потоков от размера массива
for variant, d in res[res.threads == max_t].groupby("variant"):
    d = d.sort_values("n")
    ax2.plot(d.n, d.speedup, "o-", label=variant)
ax2.axhline(1, color="gray", linestyle="--", label="без выигрыша")
ax2.set_xscale("log")
ax2.set_xlabel("размер массива n")
ax2.set_ylabel(f"ускорение при {max_t} потоках")
ax2.set_title("ускорение от размера массива")
ax2.legend()
ax2.grid(True, which="both", alpha=0.4)

fig.tight_layout()
fig.savefig(f"{stem}_speedup.png", dpi=150)