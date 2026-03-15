from sklearn.datasets import fetch_california_housing

from mlguard.experiments.experiment_manager import ExperimentManager


data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target


exp = ExperimentManager(leakage_mode="warn")

exp.fit(X, y)

print(exp.compare())