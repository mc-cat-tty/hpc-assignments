import seaborn as sb
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.ticker as mticker
import numpy as np


DF = pd.DataFrame(
  {
    "standard": [610.99, 192.88, 18.103], # ms
    "pinned": [597.27, 105.00+12.986, 9.7351], # ms
    "standard": [576.22, 98.983, 0] # ms
  },
  index = ["exec", "alloc", "copies"]
)

DF = pd.DataFrame(
  {
    "exec": [610.99, 597.27, 576.22], # ms
    "alloc": [192.88, 105.00+12.986, 98.983], # ms
    "copies": [18.103, 9.7351, 0] # ms
  },
  index = ["standard", "pinned", "UVM"]
)

def main():
  sb.set_theme(style="whitegrid")
  DF.plot(kind='bar', stacked=True, color=['red', 'skyblue', 'lightgreen'])
  
  # pl.xlabel('Memory model')
  pl.ylabel('Time [ms]')
  pl.show()

if __name__ == "__main__":
  main()
