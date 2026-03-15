from mlguard.experiments.experiment_manager import ExperimentManager
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

X = data.data
y = data.target

exp = ExperimentManager(leakage_mode="warn")

exp.fit(X,y)

print(exp.compare())