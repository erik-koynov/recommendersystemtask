import numpy as np
from typing import List, Tuple

class ModelOutput:
    def __init__(self,
                 labels: List[bool],
                 similarity_scores: np.ndarray):
        self.labels = labels
        self.similarity_scores = similarity_scores
        if len(self.labels) != len(self.similarity_scores):
            raise ValueError(f"labels and similarity scores must be of the same length, but are:"
                             f"{len(self.labels)} and {len(self.similarity_scores)} respectively.")

    def __iter__(self) -> Tuple[bool, float]:
        for label, score in zip(self.labels, self.similarity_scores):
            yield label, score

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"Similarity metric output: \nLabels: {self.labels}\nSimilarity_scores: {self.similarity_scores}"
