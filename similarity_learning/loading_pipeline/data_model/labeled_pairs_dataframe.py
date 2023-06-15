import pandas as pd
from pathlib import Path
from typing import Union, List
from ..preprocessing import JsonLoadingPipeLine
from .jobs_dataframe import JobsDataFrame
from .talents_dataframe import TalentsDataFrame
from similarity_learning.exceptions import DataLoadingError, NotSetAttributeError
from .pairs_dataframe import PairsDataFrame, IlocBase

class LabeledPairsDataFrame(PairsDataFrame):
    def __init__(self, jobs: JobsDataFrame, talents: TalentsDataFrame, labels: pd.Series, add_language_feature_suffix_flag: bool):
        super().__init__(jobs, talents, add_language_feature_suffix_flag)
        self._labels = labels
        self._iloc = Iloc(self)

    @property
    def iloc(self) -> "Iloc":
        return self._iloc

    @property
    def labels(self):
        if self._labels is None:
            raise NotSetAttributeError("labels attribute has not been set at initialization time.")
        return self._labels

    def reset_index(self, inplace: bool, drop: bool):
        jobs = self._jobs.reset_index(inplace=inplace, drop=drop)
        talents = self._talents.reset_index(inplace=inplace, drop=drop)
        labels = self._labels.reset_index(inplace=inplace, drop=drop)
        if not inplace:
            return self.__class__(jobs, talents, labels, self.add_language_feature_suffix_flag)

    def __getitem__(self, item: str) -> pd.Series:
        if not isinstance(item, str):
            raise TypeError(f"{self.__class__} indices must be strings!")
        if item == "labels":
            return self.labels
        try:
            return self._jobs[item]
        except KeyError:
            try:
                return self._talents[item]
            except KeyError:
                raise KeyError("Item was not found in any of the two datasets (neither in jobs nor in talents).")

    @classmethod
    def from_full_json(cls, file_path: Union[Path, str], add_language_suffix=False)-> "LabeledPairsDataFrame":
        loader = JsonLoadingPipeLine()
        raw_job, raw_talent, labels = loader(file_path)

        if (list(raw_job.index) != list(raw_talent.index)) or (list(raw_job.index)!=list(labels.index)):
            raise DataLoadingError("Loaded objects for jobs, talents and labels do not have the same index!")

        jobs = JobsDataFrame.from_full_json(raw_job, add_language_suffix)
        talents = TalentsDataFrame.from_full_json(raw_talent, add_language_suffix)
        return cls(jobs, talents, labels, add_language_suffix)

    def as_dataframe(self, jobs_suffix="jobs", talents_suffix="talents")->pd.DataFrame:
        return pd.concat((self._jobs.data.rename(lambda x: x + "_" + jobs_suffix, axis='columns'),
                          self._talents.data.rename(lambda x: x + "_" + talents_suffix, axis='columns'),
                          self.labels), axis=1)


class Iloc(IlocBase):

    def __getitem__(self, item: Union[int, List[int], slice]) -> LabeledPairsDataFrame:

        new_jobs = self.data._jobs.iloc[item]
        new_talents = self.data.talents.iloc[item]
        labels = self.data.labels.iloc[item]

        return LabeledPairsDataFrame(new_jobs, new_talents, labels, self.data.add_language_feature_suffix_flag)

