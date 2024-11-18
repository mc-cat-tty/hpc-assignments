import seaborn as sb
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.container import Container
import numpy as np

DF = pd.DataFrame({
  "Optimization": ["Sequential", "GPU", "Task", "Parallel for"] * 5,
  "Time":
    [0.000070, 0.162, 0.000066, 0.000268] + 
    [0.526751,  0.507421, 0.051703, 0.017396] +
    [4.916482, 2.038755, 0.464806, 0.135351] +
    [42.652974, 18.26249, 3.293214, 1.151873] +
    [460.815627, 270.2426, 26.792492, 11.367700],
  "Dataset Size": ["Mini"] * 4 + ["Small"] * 4 + ["Standard"] * 4 + ["Large"] * 4 + ["Extra Large"] * 4
})

extralarge_baseline = DF[DF["Dataset Size"] == "Extra Large"][DF["Optimization"] == "Sequential"]["Time"].item()
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
