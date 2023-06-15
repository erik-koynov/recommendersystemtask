import pandas as pd
from typing import Union, List, Dict, Any
from .jobs_dataframe import JobsDataFrame
from .talents_dataframe import TalentsDataFrame
from similarity_learning.exceptions import DataLoadingError
from .pairs_dataframe import PairsDataFrame, IlocBase


class UnlabeledPairsDataFrame(PairsDataFrame):
    def __init__(self,
                 jobs: JobsDataFrame,
                 talents: TalentsDataFrame,
                 add_language_feature_suffix_flag: bool):
        super().__init__(jobs, talents, add_language_feature_suffix_flag)
        self._iloc = Iloc(self)

    @property
    def iloc(self) -> "Iloc":
        return self._iloc

    def reset_index(self, inplace: bool, drop: bool):
        jobs = self._jobs.reset_index(inplace=inplace, drop=drop)
        talents = self._talents.reset_index(inplace=inplace, drop=drop)
        if not inplace:
            return self.__class__(jobs, talents, self.add_language_feature_suffix_flag)

    def __getitem__(self, item: str) -> pd.Series:
        if not isinstance(item, str):
            raise TypeError(f"{self.__class__} indices must be strings!")
        try:
            return self._jobs[item]
        except KeyError:
            try:
                return self._talents[item]
            except KeyError:
                raise KeyError("Item was not found in any of the two datasets (neither in jobs nor in talents).")

    @classmethod
    def from_full_json(cls,
                       raw_job: Union[pd.Series, Dict[str, Any], List[Dict[str, Any]]],
                       raw_talent: Union[pd.Series, Dict[str, Any], List[Dict[str, Any]]],
                       add_language_suffix=False) -> "UnlabeledPairsDataFrame":
        jobs = JobsDataFrame.from_full_json(raw_job, add_language_suffix)
        talents = TalentsDataFrame.from_full_json(raw_talent, add_language_suffix)

        if list(jobs.index) != list(talents.index):
            raise DataLoadingError("Loaded objects for jobs and talents do not have the same index!")

        return cls(jobs, talents, add_language_suffix)

    def as_dataframe(self, jobs_suffix="jobs", talents_suffix="talents") -> pd.DataFrame:
        return pd.concat((self._jobs.data.rename(lambda x: x + "_" + jobs_suffix, axis='columns'),
                          self._talents.data.rename(lambda x: x + "_" + talents_suffix, axis='columns')), axis=1)


class Iloc(IlocBase):

    def __getitem__(self, item: Union[int, List[int]]) -> UnlabeledPairsDataFrame:
        new_jobs = self.data._jobs.iloc[item]
        new_talents = self.data.talents.iloc[item]

        return UnlabeledPairsDataFrame(new_jobs, new_talents, self.data.add_language_feature_suffix_flag)

