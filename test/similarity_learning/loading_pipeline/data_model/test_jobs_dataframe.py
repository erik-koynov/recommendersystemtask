from unittest import TestCase
from similarity_learning.loading_pipeline.data_model import JobsDataFrame
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from dummies import job
import pandas as pd


class TestJobsDataFrame(TestCase):
    def test_from_full_json_constructor_works_with_single_dict_input(self):
        jobs_df = JobsDataFrame.from_full_json(job)
        self.assertTrue(isinstance(jobs_df.data, pd.DataFrame))
        self.assertEquals(jobs_df.index[0], 0)
        self.assertTrue(len(jobs_df.data.columns.difference(set(jobs_df.columns)))==0)
