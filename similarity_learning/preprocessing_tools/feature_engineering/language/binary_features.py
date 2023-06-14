import pandas as pd
from typing import List
import numpy as np
from ..general import intersection_length

def is_required_language_missing(title_languages_jobs: List[str],
                                 title_languages_talents: List[str],
                                 must_have_languages_jobs: List[bool])->bool:
    """
    Return binary feature whether a must_have language is missing from the talent's cv. If the intersection between his
    languages and the required languages is below the number of required languages, the feature returns True.
    :param title_languages_jobs:
    :param title_languages_talents:
    :param must_have_languages_jobs:
    :return:
    """
    required_languages = np.array(title_languages_jobs)[must_have_languages_jobs]
    return intersection_length(required_languages, title_languages_talents) < len(required_languages)


