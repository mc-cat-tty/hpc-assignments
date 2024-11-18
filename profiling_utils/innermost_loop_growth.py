import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl
from os import walk
from os.path import join
from re import compile, DOTALL

DATA_FOLDERNAME: str = "data"
DOWNSAMPLE_FACTOR: int = 1


def get_dataframes(subfolder: str, sample_type: str) -> list:
	data_frames = []
	folder = join(DATA_FOLDERNAME, subfolder)
	variable_regex = compile(r"INNERMOST_IT_VS_TIME_MS = {.+}", DOTALL)


	for data_filename in next(walk(folder), (None, None, []))[2]:
		file_path = join(folder, data_filename)
		file_content = open(file_path).read().strip()
		file_content = next(variable_regex.finditer(file_content)).group()
		exec(file_content, globals())  # Import data dictionary
		assert 'INNERMOST_IT_VS_TIME_MS' in globals(), f"Variable not properly defined in {data_filename}"

		df = pd.DataFrame({
			"Iteration": INNERMOST_IT_VS_TIME_MS.keys(),
			"Execution times [ms]": INNERMOST_IT_VS_TIME_MS.values(),
			"Sample Number": [len(data_frames)]*len(INNERMOST_IT_VS_TIME_MS),
			"Sample Type": [sample_type]*len(INNERMOST_IT_VS_TIME_MS)
		})
		df["Execution times [ms]"] = df["Execution times [ms]"].rolling(100).mean()
		data_frames.append(df[::DOWNSAMPLE_FACTOR])
	
	return data_frames


def main():
	non_opt_df = get_dataframes("non_opt", "Non Opt.")
	simd_df = get_dataframes("simd", "SIMD")

	data_frames = non_opt_df + simd_df

	data = pd.concat(data_frames, ignore_index=True)
	print(data)
	
	sb.set_theme(style="whitegrid")

	sb.lineplot(
		data=data,
		palette="rocket_r",
		hue="Sample Type",
		linewidth=2,
		x="Iteration",
		y="Execution times [ms]"
	)
	sb.despine()
	pl.show()

if __name__ == "__main__":
	main()