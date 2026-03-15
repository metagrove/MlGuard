import pandas as pd
from mlguard.inspection.leakage_detector import LeakageDetector


def test_leakage_detector():

    X = pd.DataFrame({
        "a":[1,2,3,4],
        "b":[2,4,6,8]
    })

    y = pd.Series([1,2,3,4])

    detector = LeakageDetector()

    detector.fit(X,y)

    result = detector.report()

    assert result is not None