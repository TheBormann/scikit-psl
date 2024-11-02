import logging
from collections import defaultdict
from itertools import permutations, product, chain
from typing import Optional
from sklearn.metrics import log_loss

import numpy as np
from joblib import Parallel, delayed

from .probabilistic_scoring_list import ProbabilisticScoringList
from .probabilistic_scoring_system import ProbabilisticScoringSystem

class CostSensitiveProbabilisticScoringList(ProbabilisticScoringList):
    """
    Cost-sensitive probabilistic scoring list classifier.
    Implements both greedy and Pareto front approaches for feature selection.
    """

    def __init__(
        self,
        score_set: set,
        loss_cutoff: float = None,
        method="bisect",
        selection_method="greedy",
        lookahead=1, # For greedy method
        max_iterations=10,  # For Pareto front method
        n_jobs=None,
        stage_loss=None,
        cascade_loss=None,
        stage_clf_params=None,
    ):
        """
        Initializes the CostSensitiveProbabilisticScoringList.

        :param score_set: Set of score values to be considered (feature weights).
        :param loss_cutoff: Minimal loss at which to stop fitting further stages. None means fitting all stages.
        :param method: Optimization method for threshold optimization.
        :param selection_method: Feature selection method ('greedy' or 'pareto').
        :param lookahead: Number of features to look ahead when selecting the next feature (used in greedy method).
        :param max_iterations: Maximum number of iterations for the Pareto front algorithm.
        :param n_jobs: Number of parallel jobs for computation.
        :param stage_loss: Loss function used at each stage.
        :param cascade_loss: Loss function used to aggregate losses across stages.
        :param stage_clf_params: Additional parameters for the stage classifiers.
        """
        super().__init__(
            score_set=score_set,
            loss_cutoff=loss_cutoff,
            method=method,
            lookahead=lookahead,
            n_jobs=n_jobs,
            stage_loss=stage_loss,
            cascade_loss=cascade_loss,
            stage_clf_params=stage_clf_params,
        )
        self.selection_method = selection_method
        self.max_iterations = max_iterations
        self.pareto_front = []

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        feature_costs=None,
        predef_features: Optional[np.ndarray] = None,
        predef_scores: Optional[np.ndarray] = None,
        strict=True,
    ) -> "CostSensitiveProbabilisticScoringList":
        """
        Fits a cost-sensitive probabilistic scoring list to the given data using the specified selection method.

        :param X: Feature matrix.
        :param y: Target vector.
        :param feature_costs: List or array containing the cost for each feature.
        :param predef_features: Predefined features to include.
        :param predef_scores: Predefined scores corresponding to the predefined features.
        :param strict: Whether to strictly use the predefined features.
        :return: The fitted classifier.
        """
        if self.selection_method == 'greedy':
            return self._fit_greedy(
                X, y, sample_weight, feature_costs, predef_features, predef_scores, strict
            )
        elif self.selection_method == 'pareto':
            return self._fit_pareto(
                X, y, sample_weight, feature_costs, predef_features, predef_scores, strict
            )
        else:
            raise ValueError(f"Unknown selection_method '{self.selection_method}'. Use 'greedy' or 'pareto'.")

    def _fit_greedy(
        self,
        X,
        y,
        sample_weight,
        feature_costs,
        predef_features,
        predef_scores,
        strict,
    ):
        """
        Fits the classifier using the greedy algorithm.
        """
        X, y = np.array(X), np.array(y)
        predef_features = predef_features or []
        predef_scores = predef_scores or []

        if feature_costs is None:
            raise ValueError("Feature costs must be provided.")
        if len(feature_costs) != X.shape[1]:
            raise ValueError("Length of feature_costs must equal number of features.")
        self.feature_costs = np.array(feature_costs)

        self.classes_ = np.unique(y)
        if predef_scores and predef_features:
            assert len(predef_features) <= len(predef_scores)

        predef_scores_dict = defaultdict(lambda: list(self.score_set_))
        predef_scores_dict.update({predef_features[i]: [s] for i, s in enumerate(predef_scores)})

        number_features = X.shape[1]
        remaining_features = set(range(number_features))
        self.stage_clfs = []
        self._features = []
        self._scores = []
        self._thresholds = []

        # Initial expected loss with no features
        initial_loss = self._fit_and_store_clf_at_k(X, y, sample_weight, f=[], s=[], t=[])
        losses = [initial_loss]
        stage = 0

        while remaining_features and (
            self.loss_cutoff is None or losses[-1] > self.loss_cutoff
        ):
            len_ = min(self.lookahead, len(remaining_features))
            len_pre = min(len(set(predef_features) & remaining_features), len_)
            len_rest = len_ - len_pre

            if strict and predef_features:
                prefixes = [
                    [f_ for f_ in predef_features if f_ in remaining_features][:len_pre]
                ]
            else:
                prefixes = permutations(
                    set(predef_features) & remaining_features, len_pre
                )

            f_exts = [
                list(pre) + list(suf)
                for (pre, suf) in product(
                    prefixes,
                    permutations(remaining_features - set(predef_features), len_rest),
                )
            ]

            # Collect BCR values
            bcr_values_futures = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_bcr)(
                    list(f_seq), list(s_seq), losses[-1], X, y, sample_weight
                )
                for (f_seq, s_seq) in chain.from_iterable(
                    product([fext], product(*[predef_scores_dict[f] for f in fext]))
                    for fext in f_exts
                )
            )
            bcr_values, f_list, s_list, t_list = zip(*bcr_values_futures)

            i = np.argmax(bcr_values)
            selected_feature = f_list[i]
            selected_score = s_list[i]
            selected_threshold = t_list[i]

            remaining_features.remove(selected_feature)
            self._features.append(selected_feature)
            self._scores.append(selected_score)
            self._thresholds.append(selected_threshold)

            # Fit and store the classifier with the selected feature
            new_loss = self._fit_and_store_clf_at_k(
                X,
                y,
                sample_weight,
                self._features.copy(),
                self._scores.copy(),
                self._thresholds.copy(),
            )
            losses.append(new_loss)
            stage += 1
        return self

    def _compute_bcr(
        self, feature_extension, score_extension, current_loss, X, y, sample_weight
    ):
        # Feature set with the new feature(s)
        new_features = self._features + feature_extension
        new_scores = self._scores + score_extension
        new_thresholds = self._thresholds + [None] * len(feature_extension)

        # Fit the classifier with the extended features
        clf = ProbabilisticScoringSystem(
            features=new_features,
            scores=new_scores,
            initial_feature_thresholds=new_thresholds,
            **self.stage_clf_params_,
        ).fit(X, y)
        new_loss = clf.score(X, y, sample_weight)

        # Compute the expected loss reduction
        delta_loss = current_loss - new_loss

        # Compute the cost of the new feature(s)
        feature_cost = np.sum(self.feature_costs[feature_extension])

        if feature_cost == 0:
            bcr = np.inf if delta_loss > 0 else 0
        else:
            bcr = delta_loss / feature_cost

        # Return the BCR, first feature, first score, and first threshold
        return bcr, feature_extension[0], score_extension[0], clf.feature_thresholds[-1]

    def _fit_and_store_clf_at_k(self, X, y, sample_weight=None, f=None, s=None, t=None):
        f = f or []
        s = s or []
        t = t or []

        k_clf = ProbabilisticScoringSystem(
            features=f,
            scores=s,
            initial_feature_thresholds=t,
            **self.stage_clf_params_,
        ).fit(X, y)
        self.stage_clfs.append(k_clf)
        return k_clf.score(X, y, sample_weight)

    def _fit_pareto(
        self,
        X,
        y,
        sample_weight,
        feature_costs,
        predef_features,
        predef_scores,
        strict,
    ):
        """
        Fits the classifier using the Pareto front approach.

        :param X: Feature matrix.
        :param y: Target vector.
        :param sample_weight: Sample weights.
        :param feature_costs: List or array containing the cost for each feature.
        :param predef_features: Predefined features to include.
        :param predef_scores: Predefined scores corresponding to the predefined features.
        :param strict: Whether to strictly use the predefined features.
        :return: The Pareto front, a list of non-dominated solutions.
        """
        X, y = np.array(X), np.array(y)
        predef_features = predef_features or []
        predef_scores = predef_scores or []

        if feature_costs is None:
            raise ValueError("Feature costs must be provided.")
        if len(feature_costs) != X.shape[1]:
            raise ValueError("Length of feature_costs must equal number of features.")
        self.feature_costs = np.array(feature_costs)

        self.classes_ = np.unique(y)
        if predef_scores and predef_features:
            assert len(predef_features) <= len(predef_scores)

        predef_scores_dict = defaultdict(lambda: list(self.score_set_))
        predef_scores_dict.update({predef_features[i]: [s] for i, s in enumerate(predef_scores)})

        number_features = X.shape[1]
        all_features = set(range(number_features))
        remaining_features = all_features.copy()

        # Initialize Pareto front with an empty solution
        initial_loss = self._compute_loss([], [], X, y, sample_weight)
        initial_solution = {
            'features': [],
            'scores': [],
            'thresholds': [],
            'loss': initial_loss,
            'cost': 0.0
        }
        self.pareto_front = [initial_solution]
        iteration = 0

        while iteration < self.max_iterations and remaining_features:
            new_solutions = []
            for solution in self.pareto_front:
                current_features = solution['features']
                current_scores = solution['scores']
                current_thresholds = solution['thresholds']
                used_features = set(current_features)
                available_features = remaining_features - used_features

                # Consider adding each available feature
                for feature in available_features:
                    possible_scores = predef_scores_dict.get(feature, list(self.score_set_))
                    for score in possible_scores:
                        new_features = current_features + [feature]
                        new_scores = current_scores + [score]
                        new_thresholds = current_thresholds + [None]  # Threshold to be optimized

                        # Compute loss and cost
                        loss = self._compute_loss(new_features, new_scores, X, y, sample_weight)
                        cost = solution['cost'] + self.feature_costs[feature]

                        new_solution = {
                            'features': new_features,
                            'scores': new_scores,
                            'thresholds': new_thresholds,
                            'loss': loss,
                            'cost': cost
                        }
                        new_solutions.append(new_solution)

            # Update Pareto front
            combined_solutions = self.pareto_front + new_solutions
            self.pareto_front = self._compute_pareto_front(combined_solutions)

            # Identify remaining features
            if self.pareto_front:
                features_in_solutions = [set(sol['features']) for sol in self.pareto_front]
                used_features = set.union(*features_in_solutions)
                remaining_features = all_features - used_features
            else:
                remaining_features = all_features.copy()

            # Check if loss cutoff has been reached
            min_loss = min(sol['loss'] for sol in self.pareto_front)
            if self.loss_cutoff is not None and min_loss <= self.loss_cutoff:
                break

            iteration += 1

        # Fit and store classifiers for each solution in the Pareto front
        self.pareto_classifiers = []
        for solution in self.pareto_front:
            clf = ProbabilisticScoringSystem(
                features=solution['features'],
                scores=solution['scores'],
                initial_feature_thresholds=solution['thresholds'],
                **self.stage_clf_params_,
            ).fit(X, y)
            self.pareto_classifiers.append(clf)
            # Store the classifier in the solution
            solution['classifier'] = clf

        return self.pareto_front

    def _compute_loss(self, features, scores, X, y, sample_weight):
        """
        Computes the loss for a given set of features and scores.

        :param features: List of feature indices.
        :param scores: List of corresponding scores.
        :param X: Feature matrix.
        :param y: Target vector.
        :param sample_weight: Sample weights.
        :return: Computed loss.
        """
        clf = ProbabilisticScoringSystem(
            features=features,
            scores=scores,
            initial_feature_thresholds=[None]*len(features),
            **self.stage_clf_params_,
        ).fit(X, y)
        loss = clf.score(X, y, sample_weight)
        return loss

    def _compute_pareto_front(self, solutions):
        """
        Computes the Pareto front from a list of solutions.

        :param solutions: List of solution dictionaries.
        :return: List of non-dominated solutions (Pareto front).
        """
        pareto_front = []

        for s in solutions:
            # Remove any solutions in pareto_front that are dominated by s
            pareto_front = [p for p in pareto_front if not (
                (s['loss'] <= p['loss'] and s['cost'] <= p['cost']) and
                (s['loss'] < p['loss'] or s['cost'] < p['cost'])
            )]

            # Check if s is dominated by any solution in pareto_front
            is_dominated = False
            for p in pareto_front:
                if (p['loss'] <= s['loss'] and p['cost'] <= s['cost']) and \
                (p['loss'] < s['loss'] or p['cost'] < s['cost']):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(s)

        return pareto_front