from typing import Dict, List, Any, Union, Tuple, Optional
from collections import defaultdict
import pandas as pd


def unpack_list_of_dicts_column(dataframe: pd.DataFrame,
                                column_to_unpack: str,
                                suffix='',
                                return_unpacked_column_names = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    unpacked_dataframe = json_normalize_list_of_dicts_column(dataframe[column_to_unpack])
    if suffix:
        unpacked_dataframe.rename(columns={i: i+f"_{suffix}" for i in unpacked_dataframe.columns}, inplace=True)
    new_dataframe = pd.concat((dataframe, unpacked_dataframe), axis=1)
    if return_unpacked_column_names:
        return new_dataframe, list(unpacked_dataframe.columns)
    return new_dataframe

def json_normalize_list_of_dicts_column(data: pd.Series) -> pd.DataFrame:
    unpacked_dataframe = pd.json_normalize(reformat_list_of_dicts_series(data))
    return unpacked_dataframe

def reformat_list_of_dicts_series(data: pd.Series)->pd.Series:
    return data.apply(lambda x: list_of_dicts2dict_of_lists(x))


def list_of_dicts2dict_of_lists(data: List[Dict[Any, Any]])->Dict[Any, List[Any]]:
    """
    The data in the languages section is given as a list of dictionaries.
    :param data:
    :return:
    """
    dict_of_lists = defaultdict(list)
    for dict_ in data:
        for key, value in dict_.items():
            dict_of_lists[key].append(value)

    return dict(dict_of_lists)
