from typing import Union, Dict, List, Any
import numpy as np
import pandas as pd


class FeatureHierarchyEncoder:
    def __init__(self, feature_hierarchy_mapping: Union[Dict[Any, int], List[Any]]):
        self.feature_hierarchy_mapping = feature_hierarchy_mapping
        if isinstance(feature_hierarchy_mapping, list):
            self.feature_hierarchy_mapping = self.process_list_to_index_map(self.feature_hierarchy_mapping)
        self.classes_ = list(self.feature_hierarchy_mapping.keys())

    def process_list_to_index_map(self, array: List[Any])->Dict[Any, int]:
        return {lang: i for i, lang in enumerate(array)}

    def fit(self, data: pd.Series):
        # already fitted by specifying the feature_hierarchy_mapping
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        return data.apply(self.apply_feature_hierarchy_mapping)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        return self.transform(data)

    def apply_feature_hierarchy_mapping(self, feature_value: Union[Any, List[Any]]) -> Union[int, List[int]]:
        # not handling the key error since this should be handled in the data loading or elsewhere
        if isinstance(feature_value, list):
            return [self.feature_hierarchy_mapping[value] for value in feature_value]
        return self.feature_hierarchy_mapping[feature_value]

