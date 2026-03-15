import pandas as pd


def detect_problem_type(y):

    if isinstance(y, pd.Series) is False:
        y = pd.Series(y)

    unique_values = y.nunique()

    if y.dtype == "object":
        return "classification"

    if unique_values <= 10:
        return "classification"

    return "regression"