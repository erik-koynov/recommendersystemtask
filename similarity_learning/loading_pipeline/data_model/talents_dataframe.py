from .item_dataframe_base import ItemDataFrameBase

class TalentsDataFrame(ItemDataFrameBase):
    @property
    def job_roles(self):
        return self.data.job_roles

    @property
    def salary_expectation(self):
        return self.data.salary_expectation

    @property
    def degree(self):
        return self.data.degree

    @property
    def seniority(self):
        return self.data.seniority

    @property
    def rating_languages(self):
        return self.data.rating_languages


    @property
    def title_languages(self):
        return self.data.title_languages


    @property
    def node_id(self):
        return self.data.node_id