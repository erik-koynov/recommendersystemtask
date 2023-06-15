import pandas as pd
from typing import Dict


def generate_node_ids(dataframe: pd.DataFrame, new_column_name = "node_id") -> pd.Series:
    quasi_hash = compute_quasi_hash(dataframe)
    hash_map = generate_hash_map(quasi_hash)
    ids = compute_ids(quasi_hash, hash_map)
    return ids

def compute_quasi_hash(dataframe : pd.DataFrame) -> pd.Series:
    """
    To generate node ids (job or talent) we need to find the unique sets of feature_engineering. However, a pd.Series (i.e. in this
    case a row of the pd.DataFrame) cannot be hashed, and one cannot call unique on a dataframe. To this end we encode
    the row as a concatenation of all its values as string, in order to then be able to call unique.
    :param dataframe:
    :return:
    """
    quasi_hash = dataframe.apply(lambda x: ''.join([str(i) for i in x]), axis=1)
    return quasi_hash

def generate_hash_map(quasy_hash: pd.Series) -> Dict[str, int]:
    job_hash_map = {hash_: i for i, hash_ in enumerate(quasy_hash.unique())}
    return job_hash_map

def compute_ids(quasi_hash: pd.Series, hash_map: Dict[str, int]) -> pd.Series:
    ids = quasi_hash.apply(lambda x: hash_map[x])
    return ids