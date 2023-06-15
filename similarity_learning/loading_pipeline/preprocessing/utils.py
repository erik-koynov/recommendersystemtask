from typing import Union

import pandas as pd


def inplace_option(func):
    def wrapped(data: Union[pd.DataFrame, pd.Series], *args, inplace=False, **kwargs):

        if not inplace:
            data = data.copy()

        output = func(data, *args, inplace=inplace, **kwargs)

        if not inplace:
           return output

    return wrapped
