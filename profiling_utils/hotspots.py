from typing import Dict
import seaborn as sb
import pandas as pd
import numpy as np
from time import sleep

DF = {
  "function": ["mean", "std_dev", "center_reduce", "correlation"],
  "exec_time": [0.024453, 0.024932, 0.082001, 42.776995]
}
  

def main():
  sb.set_theme(style="whitegrid")
  g = sb.catplot(
    data=DF,
    kind="bar",
    x="function",
    y="exec_time"
  )
  g.set_axis_labels("", "Execution times [ms]")
  g.legend.set_title("")
  sleep(10)

if __name__ == "__main__":
  main()