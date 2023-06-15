import pandas as pd
import numpy as np
from similarity_learning.loading_pipeline import PairsDataFrame
from ..feature_hierarchy_encoder import FeatureHierarchyEncoder
from ..language import LanguageFeatureHierarchyEncoder
from typing import Tuple, List, Union
import logging
from .language_features_order import LanguageFeaturesOrder
from sklearn.preprocessing import MultiLabelBinarizer
from similarity_learning.exceptions import NotSetAttributeError, NotFittedError
logger = logging.getLogger("similarity_learning")


class BaselineFeatureExtractor:
    """
    Order of subtraction is chosen to increase intuitiveness (more means better, negatives means insufficient).
    Language mapping order is introduced to account for some bias towards certain languages.
    """
    def __init__(self,
                 seniority_hierarchy_mapping: List[str],
                 degree_hierarchy_mapping: List[str],
                 language_rating_hierarchy_mapping: List[str],
                 language_must_have_hierarchy_mapping: List[str],
                 language_mapping_order: LanguageFeaturesOrder = LanguageFeaturesOrder.TALENTS):

        self.language_mapping_order = language_mapping_order

        self.seniority_hierarchy_encoder = FeatureHierarchyEncoder(feature_hierarchy_mapping=seniority_hierarchy_mapping)
        self.degree_hierarchy_encoder = FeatureHierarchyEncoder(feature_hierarchy_mapping=degree_hierarchy_mapping)
        self.language_rating_hierarchy_encoder = LanguageFeatureHierarchyEncoder(feature_hierarchy_mapping=language_rating_hierarchy_mapping)
        self.language_must_have_hierarchy_encoder = LanguageFeatureHierarchyEncoder(feature_hierarchy_mapping=language_must_have_hierarchy_mapping)
        self.role_encoder = MultiLabelBinarizer()

        self._seniority_features = None
        self._salary_features = None
        self._degree_features = None
        self._language_features = None
        self._role_features = None
        self.feature_names = []

        self.__fitted = False

    def fit_transform(self,
                data: PairsDataFrame,
                )->np.ndarray:
        data = data.copy()
        if not data.add_language_feature_suffix_flag: #only this format is ok
            raise NotSetAttributeError("The language_features should be preprocessed by appyling the"
                                       " add_language_feature_suffix preprocessing step on the PairsDataFrame!")

        # N x dim
        self._seniority_features = self.compute_seniority_features(data)
        self._salary_features = self.compute_salary_features(data)
        self._degree_features = self.compute_degree_features(data)
        self._language_features = self.compute_language_features(data)
        self._role_features = self.compute_role_features(data)

        self.__fitted = True
        return np.hstack((self._seniority_features,
                         self._salary_features,
                         self._degree_features,
                         self._language_features,
                         self._role_features))

    def transform(self,
                data: PairsDataFrame,
                )->np.ndarray:
        if not self.__fitted:
            raise NotFittedError("Algorithm not yet fitted in fit_transform")
        return self.fit_transform(data)

    def compute_role_features(self, data: PairsDataFrame)->np.ndarray:
        logger.info("Computing role features: difference, overall intersection.")

        encoded_roles_talent, encoded_roles_job = self.feature_hierarchy_encoding(self.role_encoder,
                                                                                  data.talents.job_roles,
                                                                                  data.jobs.job_roles)

        differences = self.binary_difference_encoding(encoded_roles_job, encoded_roles_talent)
        intersection = np.array(encoded_roles_talent*encoded_roles_job).sum(axis = 1)

        feature_names = self.feature_names_from_label_encoder("job_role_diff_", self.role_encoder) + ["job_role_intersection"]
        self.feature_names.extend(feature_names)

        return np.hstack((differences, intersection[:, None]))

    def compute_language_features(self, data: PairsDataFrame) -> np.ndarray:
        logger.info("Computing language features: rating difference and must have difference.")
        # N x L
        encoded_rating_talents, encoded_rating_jobs = self.encode_language_rating(data)

        # instead of doing redundant fitting simply reuse the classes fit by the rating encoder
        self.language_must_have_hierarchy_encoder.classes_ = self.language_rating_hierarchy_encoder.classes_
        encoded_must_have_jobs = self.language_rating_hierarchy_encoder.transform(data.jobs.must_have_languages)
        encoded_title_talents = (encoded_rating_talents>0).astype(int)

        language_rating_difference_feature = np.array(encoded_rating_talents - encoded_rating_jobs)
        language_must_have_difference_feature = self.binary_difference_encoding(encoded_must_have_jobs, encoded_title_talents)

        rating_feature_names = self.feature_names_from_label_encoder("language_rating_diff_",
                                                                     self.language_rating_hierarchy_encoder)
        must_have_feature_names = self.feature_names_from_label_encoder("language_must_have_",
                                                                     self.language_rating_hierarchy_encoder)

        self.feature_names.extend(rating_feature_names + must_have_feature_names)

        return np.hstack((language_rating_difference_feature, language_must_have_difference_feature))

    def compute_degree_features(self, data: PairsDataFrame)->np.ndarray:
        logger.info("Computing seniority features: required, available, difference.")
        encoded_degree_jobs, encoded_degree_talents = self.feature_hierarchy_encoding(self.degree_hierarchy_encoder,
                                                                                      data.jobs.min_degree,
                                                                                      data.talents.degree)
        degree_difference = encoded_degree_jobs - encoded_degree_talents

        self.feature_names.extend(['min_degree_hierarchic_job', 'degree_hierarchic_talent', "degree_diff"])
        return np.hstack((encoded_degree_jobs.values[:, None],
                         encoded_degree_talents.values[:, None],
                         degree_difference.values[:, None]))

    def compute_salary_features(self, data: PairsDataFrame)->np.ndarray:
        logger.info("Computing salary features: difference.")
        salary_difference_feature = data.jobs.max_salary.values - data.talents.salary_expectation.values

        self.feature_names.append("salary_diff")
        return salary_difference_feature[:, None]

    def compute_seniority_features(self, data: PairsDataFrame)->np.ndarray:
        logger.info("Computing seniority features:min required, max required, actual,"
                    " difference from min required, and difference from max required.")
        encoded_seniority_jobs, encoded_seniority_talents = self.feature_hierarchy_encoding(self.seniority_hierarchy_encoder,
                                                                                            data.jobs.seniorities,
                                                                                            data.talents.seniority)
        max_required_seniority = encoded_seniority_jobs.apply(max).values
        min_required_seniority = encoded_seniority_jobs.apply(min).values
        encoded_seniority_talents = encoded_seniority_talents.values

        seniority_difference_feature_max = encoded_seniority_talents - max_required_seniority
        seniority_difference_feature_min = encoded_seniority_talents - min_required_seniority

        self.feature_names.extend(["min_required_seniority", "max_required_seniority", "seniority_talents", "seniority_diff_min", "seniority_diff_max"])
        return np.hstack((min_required_seniority[:, None],
                          max_required_seniority[:, None],
                          encoded_seniority_talents[:, None],
                          seniority_difference_feature_min[:, None],
                          seniority_difference_feature_max[:, None]))


    def feature_names_from_label_encoder(self, feature_prefix: str,
                                         label_encoder: Union[FeatureHierarchyEncoder, MultiLabelBinarizer])->List[str]:
        return [feature_prefix + role for role in label_encoder.classes_]

    def binary_difference_encoding(self, a: Union[np.ndarray, pd.Series], b: Union[np.ndarray, pd.Series]):
        """
        a and b are assumed to have binary values
        :param a:
        :param b:
        :return: 0 neither required nor present, 1 present but not required, -1 required but not present, 3 required and present

        """
        binary_difference = np.array(a*2 + b)
        binary_difference[binary_difference == 2] = -1
        return binary_difference

    def feature_hierarchy_encoding(self,
                                   feature_hierarchy_encoder: Union[FeatureHierarchyEncoder, MultiLabelBinarizer],
                                   features_fit_transform: pd.Series,
                                   features_transform: pd.Series) -> Tuple[Union[pd.Series, np.ndarray],...]:
        if not self.__fitted:
            encoded_a = feature_hierarchy_encoder.fit_transform(features_fit_transform)
        else:
            encoded_a = feature_hierarchy_encoder.transform(features_fit_transform)
        encoded_b = feature_hierarchy_encoder.transform(features_transform)
        return encoded_a, encoded_b

    def encode_language_rating(self, data: PairsDataFrame)->Tuple[np.ndarray,...]:
        if self.language_mapping_order == LanguageFeaturesOrder.TALENTS:
            encoded_rating_talents, encoded_rating_jobs = self.feature_hierarchy_encoding(self.language_rating_hierarchy_encoder,
                                                                                          data.talents.rating_languages,
                                                                                          data.jobs.rating_languages)

        elif self.language_mapping_order == LanguageFeaturesOrder.JOBS:
            encoded_rating_jobs, encoded_rating_talents = self.feature_hierarchy_encoding(self.language_rating_hierarchy_encoder,
                                                                                          data.jobs.rating_languages,
                                                                                          data.talents.rating_languages)

        else:
            raise ValueError(f"The value of language_mapping_order: {self.language_mapping_order}, is not supported.")
        return encoded_rating_talents, encoded_rating_jobs