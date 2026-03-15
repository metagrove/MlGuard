from sklearn.model_selection import train_test_split

from mlguard.inspection.leakage_detector import LeakageDetector
from mlguard.utils.problem_type import detect_problem_type

from mlguard.metrics.regression_metrics import rmse
from mlguard.metrics.classification_metrics import accuracy


class ExperimentManager:

    def __init__(self, models=None, leakage_mode="warn"):

        """
        leakage_mode options:
        warn  -> print warning and continue (default)
        error -> stop experiment if leakage detected
        off   -> skip leakage detection
        """

        self.models = models
        self.leakage_mode = leakage_mode
        self.results = {}

    def fit(self, X, y):

        # -------------------------
        # Step 1 — Leakage Detection
        # -------------------------

        if self.leakage_mode != "off":

            print("Running leakage detection...")

            detector = LeakageDetector()
            detector.fit(X, y)

            report = detector.report()

            if len(report) > 0:

                if self.leakage_mode == "warn":

                    print("⚠ Potential leakage detected")
                    print(report)

                elif self.leakage_mode == "error":

                    raise ValueError(
                        "Leakage detected in dataset:\n"
                        + str(report)
                    )

            else:

                print("No leakage detected")

        else:

            print("Skipping leakage detection")

        # -------------------------
        # Step 2 — Detect Problem Type
        # -------------------------

        problem = detect_problem_type(y)

        print(f"Detected problem type: {problem}")

        # -------------------------
        # Step 3 — Load Models
        # -------------------------

        if self.models is None:

            if problem == "regression":

                from sklearn.linear_model import Ridge
                from sklearn.ensemble import RandomForestRegressor

                self.models = {
                    "ridge": Ridge(),
                    "rf": RandomForestRegressor(random_state=42)
                }

                metric = rmse

            else:

                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier

                self.models = {
                    "logistic": LogisticRegression(max_iter=1000),
                    "rf": RandomForestClassifier(random_state=42)
                }

                metric = accuracy

        else:

            # assume regression metric for user models
            metric = rmse

        # -------------------------
        # Step 4 — Train/Test Split
        # -------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------
        # Step 5 — Train Models
        # -------------------------

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            score = float(metric(y_test, pred))

            self.results[name] = score

        return self

    def compare(self):

        return sorted(self.results.items(), key=lambda x: x[1])