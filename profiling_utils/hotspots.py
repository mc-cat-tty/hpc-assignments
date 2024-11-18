import seaborn as sb
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.ticker as mticker
import numpy as np

DF = pd.DataFrame({
  "function": ["(1) mean", "(2) std_dev", "(3) center_reduce", "(4) correlation"],
  "exec_time": [0.024453, 0.024932, 0.082001, 42.776995]
})
  

def main():
  sb.set_theme(style="whitegrid")
  g = sb.catplot(
    data=DF,
    kind="bar",
    x="function",
    y="exec_time",
    alpha=.7
  )
  
  g.despine(left=True)
  g.set_axis_labels("", "Execution times [s] - log scale")
  g.legend.set_title("")
  g.ax.grid(True, which="minor")

  pl.yscale("log")
  pl.show()

if __name__ == "__main__":
  main()