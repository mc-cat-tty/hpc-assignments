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


extralarge_baseline = DF[DF["Dataset Size"] == "Extra Large"][DF["Optimization"] == "Baseline"]["Time"].item()
SPEEDUPS = [extralarge_baseline/time for time in DF[DF["Dataset Size"] == "Extra Large"]["Time"]]

def main():
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
  g.ax.bar_label(g.ax.containers[-1], labels=[f"x{speedup:.1f}" for speedup in SPEEDUPS])
  
  pl.yscale("log")
  pl.show()

if __name__ == "__main__":
  main()
