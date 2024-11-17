import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl
from os import walk
from os.path import join

DATA_FOLDERNAME: str = "data"

def main():
	data_frames = []

	i = 0
	for data_filename in next(walk(DATA_FOLDERNAME), (None, None, []))[2]:
		file_path = join(DATA_FOLDERNAME, data_filename)
		exec(open(file_path).read(), globals())  # Import data dictionary
		assert 'INNERMOST_IT_VS_TIME_US' in globals(), f"Variable not properly defined in {data_filename}"
	
		df = pd.DataFrame({
			"Iteration": INNERMOST_IT_VS_TIME_US.keys(),
			"Execution times [us]": INNERMOST_IT_VS_TIME_US.values(),
			"Sample Number": [i]*len(INNERMOST_IT_VS_TIME_US)
		})
		df["Execution times [us]"] = df["Execution times [us]"].rolling(100).mean()
		data_frames.append(df)
		i += 1

	data = pd.concat(data_frames, ignore_index=True)
	print(data)
	
	sb.set_theme(style="whitegrid")

	sb.lineplot(
		data=data,
		palette="tab10",
		hue="Sample Number",
		linewidth=2,
		x="Iteration",
		y="Execution times [us]",
		# errorband='sd'
	)

	pl.show()

if __name__ == "__main__":
	main()