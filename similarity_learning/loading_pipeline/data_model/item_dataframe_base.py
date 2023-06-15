import pandas as pd
from ..preprocessing.reformat_dict_fields import unpack_list_of_dicts_column
from ..preprocessing.elementwise_sorting import element_wise_sorting, ElementWiseSortingDType
from ..preprocessing.retrieve_class_properties import retrieve_class_properties
from similarity_learning.exceptions import DataLoadingError
from ..preprocessing.node_id_generation import generate_node_ids
from typing import Union, List, Dict, Any
from abc import ABC, abstractmethod
from ..preprocessing.expand_raw_categorical_features import add_language_suffix
from similarity_learning.exceptions import DoubleUsageError

class ItemDataFrameBase(ABC):
    def __init__(self, data: pd.DataFrame, add_language_suffix=False):
        self.data = data
        self.columns = retrieve_class_properties(self.__class__)
        column_diff = set(self.data.columns).symmetric_difference(set(self.columns))
        if  len(column_diff) != 0:
            raise DataLoadingError(f"the loaded data columns: {self.data.columns} are different form the required data "
                                   f"properties: {self.columns}. The difference is {column_diff}.")

        self.iloc = Iloc(self)
        self.__is_add_language_suffix_applied = False

        if add_language_suffix:
            self.add_language_suffix_to_raw_language_features()


    def __repr__(self):
        return repr(self.data)

    def reset_index(self, inplace: bool, drop: bool):
        data = self.data.reset_index(inplace=inplace, drop=drop)
        if not inplace:
            return self.__class__(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item)->pd.Series:
        if item in self.columns:
            return self.data[item]
        else:
            raise KeyError(f"No {item} key in data.")


    @property
    def index(self):
        return self.data.index

    @property
    def languages_(self):
        return list(self.title_languages.explode().unique())

    @classmethod
    def from_full_json(cls,
                       data: Union[pd.Series, Dict[str, Any], List[Dict[str, Any]]],
                       add_language_suffix=False)->"ItemDataFrameBase":
        data: pd.DataFrame = pd.json_normalize(data)
        data.languages = element_wise_sorting(data.languages, ElementWiseSortingDType.DICT)
        data, unpacked_column_names = unpack_list_of_dicts_column(data, 'languages',
                                                                  suffix='languages',
                                                                  return_unpacked_column_names=True)
        data = data.drop(columns=["languages"])
        data.loc[:, "node_id"] = generate_node_ids(data)

        return cls(data, add_language_suffix)

    @property
    @abstractmethod
    def rating_languages(self) -> pd.Series:
        pass

    @property
    @abstractmethod
    def title_languages(self) -> pd.Series:
        pass

    @property
    @abstractmethod
    def node_id(self) -> pd.Series:
        pass

    def add_language_suffix_to_raw_language_features(self):
        if self.__is_add_language_suffix_applied:
            raise DoubleUsageError("This operation has already been applied!")
        self.data = add_language_suffix(self.data,
                                        ["rating_languages", "must_have_languages"],
                                        language_title_column_name="title_languages")
        self.__is_add_language_suffix_applied = True


class Iloc:
    def __init__(self, data: ItemDataFrameBase):
        self.data = data

    def __getitem__(self, item: Union[int, List[int], slice]) -> ItemDataFrameBase:
        if isinstance(item, int):
            new_data = pd.DataFrame(self.data.data.iloc[item]).T
        elif isinstance(item, list) or isinstance(item, slice):
            new_data = self.data.data.iloc[item]
        else:
            raise ValueError("item must be of type int or a list of ints")

        return self.data.__class__(new_data)
