from .item_dataframe_base import ItemDataFrameBase

class JobsDataFrame(ItemDataFrameBase):

    @property
    def job_roles(self):
        return self.data.job_roles

    @property
    def max_salary(self):
        return self.data.max_salary

    @property
    def min_degree(self):
        return self.data.min_degree

    @property
    def seniorities(self):
        return self.data.seniorities

    @property
    def rating_languages(self):
        return self.data.rating_languages
    @property
    def title_languages(self):
        return self.data.title_languages

    @property
    def must_have_languages(self):
        return self.data.must_have_languages

    @property
    def node_id(self):
        return self.data.node_id