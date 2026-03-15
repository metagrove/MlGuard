from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from mlguard.experiments.experiment_manager import ExperimentManager


def test_experiment_manager():

    X,y = make_regression(n_samples=100,n_features=5)

    models = {
        "ridge":Ridge(),
        "rf":RandomForestRegressor()
    }

    exp = ExperimentManager(models)

    exp.fit(X,y)

    results = exp.compare()

    assert len(results) > 0