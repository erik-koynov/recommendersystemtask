import pandas as pd
from pathlib import Path
from typing import Union, List
from ..utils import JsonLoadingPipeLine
from .jobs_dataframe import JobsDataFrame
from .talents_dataframe import TalentsDataFrame
import pickle
from ..utils.exceptions import DataLoadingError
from copy import deepcopy

class PairsDataFrame:
    def __init__(self, jobs: JobsDataFrame, talents: TalentsDataFrame, labels: pd.Series, add_language_feature_suffix_flag: bool):
        self._jobs = jobs
        self._talents = talents
        self._labels = labels
        self.add_language_feature_suffix_flag = add_language_feature_suffix_flag
        self.iloc = Iloc(self)

    @property
    def index(self):
        return self._jobs.index

    @property
    def jobs(self):
        return self._jobs

    @property
    def talents(self):
        return self._talents

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.labels)

    def copy(self):
        return deepcopy(self)

    def reset_index(self, inplace: bool, drop: bool):
        jobs = self._jobs.reset_index(inplace=inplace, drop=drop)
        talents = self._talents.reset_index(inplace=inplace, drop=drop)
        labels = self._labels.reset_index(inplace=inplace, drop=drop)
        if not inplace:
            return self.__class__(jobs, talents, labels, self.add_language_feature_suffix_flag)

    def __getitem__(self, item: str) -> pd.Series:
        """
        Search for a property sequentially - first in jobs, then in talents.
        :param item:
        :return:
        """
        if not isinstance(item, str):
            raise TypeError(f"{self.__class__} indices must be strings!")
        if item=="labels":
            return self.labels
        try:
            return self.jobs[item]
        except KeyError:
            try:
                return self.talents[item]
            except KeyError:
                raise KeyError("Item was not found in any of the two datasets (neither in jobs nor in talents).")

    @classmethod
    def from_full_json(cls, file_path: Union[Path, str], add_language_suffix=False)->"PairsDataFrame":
        loader = JsonLoadingPipeLine()
        raw_job, raw_talent, labels = loader(file_path)

        if (list(raw_job.index) != list(raw_talent.index)) or (list(raw_job.index)!=list(labels.index)):
            raise DataLoadingError("Loaded objects for jobs, talents and labels do not have the same index!")

        jobs = JobsDataFrame.from_full_json(raw_job, add_language_suffix)
        talents = TalentsDataFrame.from_full_json(raw_talent, add_language_suffix)
        return cls(jobs, talents, labels, add_language_suffix)

    def as_dataframe(self, jobs_suffix="jobs", talents_suffix="talents")->pd.DataFrame:
        return pd.concat((self.jobs.data.rename(lambda x: x + "_" + jobs_suffix, axis='columns'),
                          self.talents.data.rename(lambda x: x + "_" + talents_suffix, axis='columns'),
                          self.labels), axis=1)

    def to_pickle(self, file_path: Union[Path, str]):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_path: Union[Path, str]):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"Loaded data is not of type {cls}")
        return data


    def __repr__(self):
        return f"{self.__class__} object with {len(self)} elements. \nJobs: {self.jobs.columns}, " \
               f"\nTalents: {self.talents.columns}, \nLabels: pd.Series object."

    def add_language_feature_suffix(self):
        if not self.add_language_feature_suffix_flag:
            self.jobs.add_language_suffix_to_raw_language_features()
            self.talents.add_language_suffix_to_raw_language_features()
            self._add_language_feature_suffix = True

class Iloc:
    def __init__(self, data: PairsDataFrame):
        self.data = data

    def __getitem__(self, item: Union[int, List[int]]) -> PairsDataFrame:

        new_jobs = self.data.jobs.iloc[item]
        new_talents = self.data.talents.iloc[item]
        labels = self.data.labels.iloc[item]

        return PairsDataFrame(new_jobs, new_talents, labels)

