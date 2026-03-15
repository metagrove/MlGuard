import pandas as pd

class LeakageDetector:

    def __init__(self, threshold=0.9):

        self.threshold = threshold

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.corr_ = X.corrwith(y)
        return self

    def report(self):

        suspicious = self.corr_[abs(self.corr_) > self.threshold]
        return suspicious