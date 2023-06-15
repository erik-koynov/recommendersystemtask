from similarity_learning.loading_pipeline import LabeledPairsDataFrame
from pathlib import Path

raw_data_path = Path(__file__).parent.parent / Path("data") / Path("raw_data.json")
output_file = raw_data_path.parent / Path("raw_data_as_dataframe.pkl")
pickle_output_file = output_file.parent / Path("pairs_dataframe_object.pkl")

print(f"Loading data from: {raw_data_path}")

df = LabeledPairsDataFrame.from_full_json(raw_data_path, add_language_suffix=True)

print(f"Saving data to pickle file: {output_file}")
df.as_dataframe().to_pickle(output_file)

print(f"Saving data to pickle file: {pickle_output_file}")
df.to_pickle(pickle_output_file)