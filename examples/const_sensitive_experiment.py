from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from skpsl import CostSensitiveProbabilisticScoringList

if __name__ == '__main__':
    # Create a dataset with features of varying importance
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,  # First 5 features are informative
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Costs are assigned inversely to importance, so informative features have lower costs.
    # Features 0-4 are informative; others are not
    feature_importance = np.array([5 - i if i < 5 else 0 for i in range(X.shape[1])])

    # Normalize importance to sum to 1 (to avoid division by zero)
    feature_importance = feature_importance / feature_importance.sum()

    # Assign costs inversely proportional to importance
    # Add a small value to avoid division by zero
    feature_costs = 1 / (feature_importance + 1e-5)

    # Print out feature costs
    print("Feature costs:")
    for i, cost in enumerate(feature_costs):
        print(f"Feature {i}: Cost = {cost:.4f}")

    # Initialize and fit the classifier
    clf = CostSensitiveProbabilisticScoringList(
        score_set={-1, 1, 2}
    )
    
    clf.fit(X_train, y_train, feature_costs=feature_costs)

    # Evaluate on the test set
    brier_score = clf.score(X_test, y_test)
    print(f"\nBrier score on test set: {brier_score:.4f}")

    # Print out the selected features and their costs
    print("\nSelected features and their costs:")
    for feature_index in clf.features:
        cost = feature_costs[feature_index]
        print(f"Feature {feature_index}: Cost = {cost:.4f}")

    df = clf.inspect()
    print("\nModel inspection:")
    print(df.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))