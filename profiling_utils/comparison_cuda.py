import seaborn as sb
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.container import Container
import numpy as np

DF = pd.DataFrame({
  "Optimization": ["Baseline", "OMP", "CUDA"] * 5,
  "Time":
    [0.000057,0.00924141,0.0836763] +
    [0.562221,0.0214025,0.10178] +
    [4.96188,0.172975,0.183133] +
    [43.6303,1.28537,0.695554] +
    [455.688,11.7039,4.32089],
  "Dataset Size": ["Mini"] * 3 + ["Small"] * 3 + ["Standard"] * 3 + ["Large"] * 3 + ["Extra Large"] * 3
})

SPEEDUPS = []
for size in DF["Dataset Size"].unique():
  baseline = DF[DF["Dataset Size"] == size][DF["Optimization"] == "Baseline"]["Time"].item()
  SPEEDUPS.append([baseline/time for time in DF[DF["Dataset Size"] == size]["Time"]])

def global_comparison():
  sb.set_theme(style="whitegrid")
  g = sb.catplot(
    data=DF,
    kind="bar",
    x="Optimization",
    y="Time",
    hue="Dataset Size",
    palette="dark",
    alpha=.7
  )
  
  g.despine(left=True)
  g.set_axis_labels("", "Time [s] - log scale")
  g.legend.set_title("")
  g.ax.grid(True, which="minor")
  for container, labels in zip(g.ax.containers, SPEEDUPS):
    g.ax.bar_label(container, labels=[f"x{speedup:.1f}" for speedup in labels])
  
  pl.yscale("log")
  pl.show()

def large_dataset_comparison():
  sb.set_theme(style="whitegrid")
  g = sb.catplot(
    data=DF[DF["Dataset Size"] == "Large"],
    kind="bar",
    x="Optimization",
    y="Time",
    hue="Dataset Size",
    palette="dark",
    alpha=.7,
  )
  
  g.despine(left=True)
  g.set_axis_labels("", "Time [s]")
  g.legend.set_title("")
  g.ax.grid(True, which="minor")
  g.ax.bar_label(g.ax.containers[0], labels=[f"x{speedup:.1f}" for speedup in SPEEDUPS[-2]])
  
  # DF2 = pd.concat([
  #   DF[DF["Dataset Size"] == "Large"],
  #   pd.DataFrame({"Speedup": [s for s in SPEEDUPS[-2]], "Optimization": ["Baseline", "OMP", "CUDA"]})
  # ])

  # h = sb.lineplot(
  #   data=DF2,
  #   x="Optimization",
  #   y="Speedup",
  #   palette="dark",
  #   alpha=.7,
  #   legend=False,
  # )
  # h.set_ylim((min(SPEEDUPS[-2]), max(SPEEDUPS[-2])))

  pl.yscale("linear")
  pl.show()

def main():
  global_comparison()
  large_dataset_comparison()

if __name__ == "__main__":
  main()
