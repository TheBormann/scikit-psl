
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from skpsl import CostSensitiveProbabilisticScoringList

if __name__ == '__main__':
    X, y = make_classification(n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_costs = np.random.rand(X.shape[1])

    clf = CostSensitiveProbabilisticScoringList(
        score_set={-1, 1, 2}
    )
    
    clf.fit(X_train, y_train, feature_costs=feature_costs)

    brier_score = clf.score(X_test, y_test)
    print(f"Brier score: {brier_score:.4f}")

    df = clf.inspect(5)
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))