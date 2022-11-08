"""

Add Ending X, Y from the Target and Predicted Index to Model Output Excel file.

The resulting file can then be used to analyze ending index (0-99) and if the X, Y variables
are also on the edges like we've seen on the initial conditions - X, Y.

Date Created: 11/8/2022

"""

import pandas as pd
# requires "openpyxl" dependency


INPUT_FILENAME = "../model_output/PyTorch_SGD_BestModel__09-19-2022_163431.xlsx"
OUTPUT_FILENAME = "../model_output/PyTorch_SGD_EndingConditions__09-19-2022_163431.xlsx"


def add_ending_conditions(df: pd.DataFrame, col: str) -> pd.DataFrame:
	"""Add ending X, Y conditions from a given index column in the 10x10 grid."""
	df[f"{col}_X"] = df[col] % 10  # zeroes digit  24 -> 2(4) -> 4
	df[f"{col}_Y"] = df[col] // 10  # tens digit  24 -> (2)4 -> 2
	return df


def main() -> int:
	data = pd.read_excel(INPUT_FILENAME, sheet_name='RawResults')
	print(data.head())
	print(data.info())
	convert_cols = ['Target', 'Prediction']

	for col in convert_cols:
		data = add_ending_conditions(df=data, col=col)

	data.to_excel(OUTPUT_FILENAME, index=False)


if __name__ == '__main__':
	main()