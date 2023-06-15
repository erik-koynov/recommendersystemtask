from unittest import TestCase

import numpy as np

from similarity_learning.inference.baseline_model.decision_tree_model import DecisionTreeModel
from similarity_learning.loading_pipeline import LabeledPairsDataFrame
from pathlib import Path

class TestDecisionTreeModel(TestCase):
    def setUp(self) -> None:
        stored_models_path = Path(__file__).parent.parent.parent.parent.parent / Path("stored_models")
        data_path = stored_models_path.parent / Path("data") / Path("pairs_dataframe_object.pkl")

        self.tree_sklearn_path = stored_models_path / Path("decision_tree_sklearn.pkl")
        self.tree_model_path = stored_models_path / Path("decision_tree_model_object.pkl")

        self.pairs_dataframe = LabeledPairsDataFrame.from_pickle(data_path)

    def test_loads_a_decision_tree_model_object_from_pickle_file(self):
        model = DecisionTreeModel.load(self.tree_model_path)
        self.assertTrue(isinstance(model, DecisionTreeModel))

    def test_throws_exception_if_pickle_file_stores_object_of_wrong_type(self):
        self.assertRaises(TypeError, lambda : DecisionTreeModel.load(self.tree_sklearn_path))

    def test_predict_outputs_object_of_the_same_length_as_the_input(self):
        model = DecisionTreeModel.load(self.tree_model_path)
        preds = model.predict(self.pairs_dataframe.iloc[0:10])
        self.assertEquals(len(preds), 10)
        preds = model.predict(self.pairs_dataframe.iloc[0])
        self.assertEquals(len(preds), 1)

    def test_compute_similarity_score_returns_squeezed_array(self):
        model = DecisionTreeModel.load(self.tree_model_path)
        score = model.compute_similarity_score(np.random.rand(10, 3))
        self.assertEquals(score.shape, (10, ))


