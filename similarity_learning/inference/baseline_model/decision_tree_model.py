import numpy as np
from ..model import Model
from sklearn.base import BaseEstimator
from similarity_learning.feature_engineering import BaselineFeatureExtractor
from ..model_output import ModelOutput
from ...loading_pipeline import PairsDataFrame
from typing import List


class DecisionTreeModel(Model):
    def __init__(self, model: BaseEstimator, feature_extractor: BaselineFeatureExtractor):
        super().__init__(model)
        self.feature_extractor = feature_extractor

    def predict(self, data: PairsDataFrame) -> ModelOutput:
        features = self.feature_extractor.transform(data)

        scores: np.ndarray = self.model.predict_proba(features)
        similarity_scores = self.compute_similarity_score(scores)
        labels: List[bool] = self.label_assignment(similarity_scores)

        return ModelOutput(labels, similarity_scores)

    def compute_similarity_score(self, scores: np.ndarray) -> np.ndarray:
        return scores[:, 1]