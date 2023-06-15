from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Union, List
import torch.nn as nn
from ..loading_pipeline import PairsDataFrame
from .model_output import ModelOutput
import numpy as np
from pathlib import Path
import pickle

class Model(ABC):
    def __init__(self, model: Union[BaseEstimator, nn.Module]):
        self.model = model

    @classmethod
    def load(cls, file_path: Union[Path, str]) -> "Model":
        with open(file_path, 'rb') as f:
            model = pickle.load(f)

        if isinstance(model, cls):
            return model

        raise TypeError(f"loaded model is not of type {cls}.")


    def save(self, file_path: Union[Path, str]):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def predict(self, data: PairsDataFrame) -> ModelOutput:
        """
        Predict on PairsDataFrame objects.
        :param data:
        :return:
        """

    def label_assignment(self, similarity_scores: np.ndarray) -> List[bool]:
        if len(similarity_scores.shape) != 1:
            raise ValueError("Similarity score per batch element should be a single float literal.")
        return [bool(i) for i in similarity_scores]

    @abstractmethod
    def compute_similarity_score(self, score: np.ndarray) -> np.ndarray:
        """
        Provide a similarity score from the computed score
        :param score:
        :return:
        """