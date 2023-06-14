from typing import List, Any

def intersection_length(a: List[Any], b: List[Any]):
    """
    Compute the intersection length between two features, that are represented as list of attributes
    :param a:
    :param b:
    :return:
    """
    return len(set(a).intersection(set(b)))