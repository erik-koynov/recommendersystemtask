import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple
from similarity_learning.globals import JSON_FORMAT_FIELDS
from .exceptions import UnexpectedJsonFormatException



class JsonLoadingPipeLine:
    def __init__(self, high_level_keys: List[str] = None):
        if high_level_keys is None:
            self.high_level_keys = JSON_FORMAT_FIELDS
        else:
            self.high_level_keys = high_level_keys

    def __call__(self, file_path: Union[Path, str]) -> Tuple[pd.Series, ...]:
        raw_data = self._load_full_json(file_path)
        return self._split_into_constituent_dataframes(raw_data)

    def _load_full_json(self, file_path: Union[Path, str])->pd.DataFrame:
        raw_data = pd.read_json(file_path)
        if set(raw_data.columns) != set(self.high_level_keys):
            raise UnexpectedJsonFormatException(f"loaded data from json file does not have the expected "
                                                f"columns: {self.high_level_keys}, and has {list(raw_data.columns)} instead.")
        return raw_data


    def _split_into_constituent_dataframes(self, raw_data: pd.DataFrame) -> Tuple[pd.Series,...]:
        return tuple([raw_data[key] for key in self.high_level_keys])
