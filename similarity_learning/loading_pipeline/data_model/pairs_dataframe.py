from abc import ABC, abstractmethod
import pandas as pd
from .jobs_dataframe import JobsDataFrame
from .talents_dataframe import TalentsDataFrame
from typing import Union, List
import pickle
from pathlib import Path
from copy import deepcopy


class PairsDataFrame(ABC):
    def __init__(self, jobs: JobsDataFrame, talents: TalentsDataFrame, add_language_feature_suffix_flag: bool):
        self._jobs = jobs
        self._talents = talents
        self.add_language_feature_suffix_flag = add_language_feature_suffix_flag


    @classmethod
    @abstractmethod
    def from_full_json(cls, *args, **kwargs) -> "PairsDataFrame":
        """
        Load data from objects which are assumed to be in the format provided in the task
        :param file_path:
        :param add_language_suffix:
        :return:
        """

    @abstractmethod
    def reset_index(self, inplace: bool, drop: bool):
        """
        Mimic the behaviour of pandas df reset index. For all data attributes.
        :param inplace:
        :param drop:
        :return:
        """

    @abstractmethod
    def __getitem__(self, item: str) -> pd.Series:
        """

        :param item:
        :return:
        """

    @property
    @abstractmethod
    def iloc(self)->"IlocBase":
        pass

    @property
    def index(self):
        return self._jobs.index

    @property
    def jobs(self):
        return self._jobs

    @property
    def talents(self):
        return self._talents

    @classmethod
    def from_pickle(cls, file_path: Union[Path, str]):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"Loaded data is not of type {cls}")
        return data

    def add_language_feature_suffix(self):
        if not self.add_language_feature_suffix_flag:
            self._jobs.add_language_suffix_to_raw_language_features()
            self._talents.add_language_suffix_to_raw_language_features()
            self.add_language_feature_suffix_flag = True

    def copy(self):
        return deepcopy(self)

    def to_pickle(self, file_path: Union[Path, str]):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def as_dataframe(self, jobs_suffix="jobs", talents_suffix="talents") -> pd.DataFrame:
        """
        Concatenate the data attributes into one dataframe.
        :param jobs_suffix: suffix to add to all jobs columns (as the column names often overlap due to the common base class)
        :param talents_suffix: (see above)
        :return:
        """

    def __repr__(self):
        return f"{self.__class__} object with {len(self)} elements. \nJobs: {self._jobs.columns}, " \
               f"\nTalents: {self._talents.columns}."

    def __len__(self):
        return len(self._jobs)


class IlocBase(ABC):
    def __init__(self, data: PairsDataFrame):
        self.data = data

    @abstractmethod
    def __getitem__(self, item: Union[int, List[int]]) -> PairsDataFrame:
        """
        Call the iloc method on the data attributes of the PairsDataFrameBase objects
        :param item:
        :return:
        """