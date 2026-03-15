from .base_estimator import BaseEstimator


class Transformer(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):

        self.fit(X, y)

        return self.transform(X)