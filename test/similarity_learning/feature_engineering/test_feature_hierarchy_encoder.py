from unittest import TestCase

import numpy as np
import pandas as pd
from similarity_learning.feature_engineering import FeatureHierarchyEncoder

class TestFeatureHierarchyEncoder(TestCase):
    def setUp(self) -> None:
        self.dummy_data_single_value = pd.Series([1, 2, 1, 2, 3, 3, 3, 1])
        self.dummy_data_multiple_values = pd.Series([[1,2,3],[1],[3,2],[1,3]])
        self.encoder = FeatureHierarchyEncoder(feature_hierarchy_mapping=[1,2,3])

    def test_maps_the_provided_hierarchy_map_to_dict_where_the_keys_are_the_items_of_the_parameter(self):
        self.assertEquals(self.encoder.feature_hierarchy_mapping, {1: 0, 2: 1, 3: 2})

    def test_encodes_single_valued_series_as_series_of_the_same_length(self):
        out = self.encoder.transform(self.dummy_data_single_value)
        self.assertTrue(len(out)==len(self.dummy_data_single_value))
        self.assertTrue((out+1).equals(self.dummy_data_single_value))

    def test_encodes_multi_valued_series_as_series_of_the_same_shape(self):
        out = self.encoder.transform(self.dummy_data_multiple_values)
        self.assertTrue(out.shape == self.dummy_data_multiple_values.shape)
        self.assertEquals([i+1 for i in out.iloc[0]], self.dummy_data_multiple_values.iloc[0])


