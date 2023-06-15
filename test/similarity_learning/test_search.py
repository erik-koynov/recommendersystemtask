from similarity_learning.search import Search
from similarity_learning.inference import DecisionTreeModel
from unittest import TestCase
from pathlib import Path
from dummies import talent, job
from itertools import chain

class TestSearch(TestCase):
    def setUp(self) -> None:
        data_path = Path(__file__).parent.parent.parent / Path("stored_models") / Path("decision_tree_model_object.pkl")
        model = DecisionTreeModel.load(data_path)
        self.search = Search(model)

    def test_returns_dict_with_correct_keys_and_values_given_single_input(self):
        out = self.search.match(talent=talent, job = job)

        self.assertTrue(isinstance(out, dict))
        self.assertEquals(list(out.keys()), ["talent", "job", "label", "score"])
        self.assertTrue(isinstance(out['label'], bool))
        self.assertTrue(isinstance(out["score"], float))

    def test_returns_list_of_dicts_with_correct_keys_and_values_given_batch_input(self):
        out = self.search.match_bulk(talents=[talent]*10, jobs=[job]*10)
        self.assertTrue(isinstance(out, list))
        keys = list(chain(*[o.keys()for o in out]))
        self.assertTrue(len(set(keys).symmetric_difference({"talent", "job", "label", "score"})) == 0)


