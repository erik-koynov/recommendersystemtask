from typing import List
import pandas as pd
from similarity_learning.preprocessing_tools.utils import inplace_option

def _add_language_suffix(titles_language: List[str], other_language_feature: List[str]):
    """
    To be able to easily create a one-hot or integer encoded features for the different language descriptors
    such as must_have or rating it will be best to add the language name as suffix as preprocessing step.
    :param titles_language:
    :param other_language_feature:
    :return:
    """
    return [(str(other) + "_" + str(title)) for title, other in zip(titles_language, other_language_feature)]


def add_language_suffix_on_series(titles_language: pd.Series, other_language_feature: pd.Series)->pd.Series:
    return pd.concat((titles_language, other_language_feature), axis=1).apply(lambda x: _add_language_suffix(*x), axis=1)

@inplace_option
def add_language_suffix(data, language_column_names: List[str], language_title_column_name: str, inplace=False):
    for column in set(language_column_names).difference(set([language_title_column_name])):
        try:
            data.loc[:, column] = add_language_suffix_on_series(data[language_title_column_name], data[column])
        except KeyError:
            pass
    return data

