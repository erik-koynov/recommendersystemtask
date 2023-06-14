from enum import Enum
import pandas as pd


class ElementWiseSortingDType(Enum):
    DICT = 1
    LIST = 2

def element_wise_sorting(data: pd.Series, dtype: ElementWiseSortingDType, dict_key_for_sorting = 'title'):
    if dtype == ElementWiseSortingDType.DICT:
        return data.apply(lambda x: sorted(x, key = lambda y: y[dict_key_for_sorting]))
    if dtype == ElementWiseSortingDType.LIST:
        return data.apply(sorted)
    else:
        ValueError(f"Unsupported ElementWiseSortingDType: {dtype}.")