import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from ..feature_hierarchy_encoder import FeatureHierarchyEncoder

class LanguageFeatureHierarchyEncoder(FeatureHierarchyEncoder):
    def __init__(self, feature_hierarchy_mapping: Union[Dict[str, int], list], language_list: List[str]=None):
        super().__init__(feature_hierarchy_mapping)
        self.classes_ = language_list
        if self.classes_ is not None:
            self.language_idx_map = self.process_list_to_index_map(self.classes_)
        else:
            self.language_idx_map = None


    def fit(self, data: pd.Series):
        """
        It is expected that an entry of the data consists of a List of strings encoded in the following manner
        "<feature>_<language>"
        :param data:
        :return:
        """
        if self.language_idx_map is None:
            data = self._separate_features_and_languages(data)
            self._fit_on_separated_features_and_languages(data)

    def fit_transform(self, data: pd.Series)->np.ndarray:
        """
        It is expected that an entry of the data consists of a List of strings encoded in the following manner
        :param data:
        :return:
        """
        data = self._separate_features_and_languages(data)
        self._fit_on_separated_features_and_languages(data)
        feature_matrix = self.compute_feature_matrix(data)
        return feature_matrix

    def transform(self, data: pd.Series)->np.ndarray:
        """
        It is expected that an entry of the data consists of a List of strings encoded in the following manner
        "<feature>_<language>"
        :param data:
        :return:
        """
        data = self._separate_features_and_languages(data)
        feature_matrix = self.compute_feature_matrix(data)
        return feature_matrix


    def _fit_on_separated_features_and_languages(self, data: pd.Series):
        """
        Data is in the format List [Tuple[str], Tuple[str]], where the tuples are the separated features/ languages
        :param data:
        :return:
        """
        self.classes_ = sorted(list(data.apply(lambda x: x[1]).explode().unique()))
        self.language_idx_map = self.process_list_to_index_map(self.classes_)


    def compute_feature_matrix(self, data):
        """
        Data is in the format List [Tuple[str], Tuple[str]], where the tuples are the separated features/ languages
        :param data:
        :return:
        """
        feature_series = data.apply(lambda x: self.compute_feature_vector(*x))
        return np.vstack(feature_series)

    def compute_feature_vector(self, features, languages):
        vector = np.zeros(len(self.language_idx_map))
        for feature, language in zip(features, languages):

            try:
                vector[self.language_idx_map[language]] = self.apply_feature_hierarchy_mapping(feature)
            except KeyError:
                # if language is not in the language index map or the language hierarchy map-> don't update the vector
                pass

        return vector

    def _separate_features_and_languages(self, data: pd.Series):
        data = data.copy()
        data = data.apply(self._separate_features_and_languages_in_row)
        return data

    def _separate_features_and_languages_in_row(self, data_row: List[str]) -> List[Tuple[str]]:
        """
        The initial data is a list of "feature_language", now it will be separate lists of features and languages
        :param data_row:
        :return:
        """
        return list(zip(*[element.split("_") for element in data_row]))