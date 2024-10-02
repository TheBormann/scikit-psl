import logging
from collections import defaultdict
from itertools import permutations, product, chain
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from .probabilistic_scoring_list import ProbabilisticScoringList
from .probabilistic_scoring_system import ProbabilisticScoringSystem

class CostSensitiveProbabilisticScoringList(ProbabilisticScoringList):
    """
    Cost-sensitive probabilistic scoring list classifier.
    Implements the greedy algorithm using Benefit-Cost Ratio (BCR).
    """

    def __init__(
        self,
        score_set: set,
        loss_cutoff: float = None,
        method="bisect",
        lookahead=1,
        n_jobs=None,
        stage_loss=None,
        cascade_loss=None,
        stage_clf_params=None,
        feature_costs=None,
    ):
        """
        Initializes the CostSensitiveProbabilisticScoringList with feature costs.

        :param score_set: Set of score values to be considered (feature weights).
        :param loss_cutoff: Minimal loss at which to stop fitting further stages. None means fitting all stages.
        :param method: Optimization method for threshold optimization.
        :param lookahead: Number of features to look ahead when selecting the next feature.
        :param n_jobs: Number of parallel jobs for computation.
        :param stage_loss: Loss function used at each stage.
        :param cascade_loss: Loss function used to aggregate losses across stages.
        :param stage_clf_params: Additional parameters for the stage classifiers.
        :param feature_costs: List or array containing the cost for each feature.
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
        self.feature_costs = feature_costs

    def fit(
    self,
    X,
    y,
    sample_weight=None,
    predef_features: Optional[np.ndarray] = None,
    predef_scores: Optional[np.ndarray] = None,
    strict=True,
    ) -> "CostSensitiveProbabilisticScoringList":
        """
        Fits a cost-sensitive probabilistic scoring list to the given data.

        :param X: Feature matrix.
        :param y: Target vector.
        :param predef_features: Predefined features to include.
        :param predef_scores: Predefined scores corresponding to the predefined features.
        :param strict: Whether to strictly use the predefined features.
        :return: The fitted classifier.
        """
        X, y = np.array(X), np.array(y)
        predef_features = predef_features or []
        predef_scores = predef_scores or []

        if self.feature_costs is None:
            raise ValueError("Feature costs must be provided.")
        if len(self.feature_costs) != X.shape[1]:
            raise ValueError("Length of feature_costs must equal number of features.")

        self.classes_ = np.unique(y)
        if predef_scores and predef_features:
            assert len(predef_features) <= len(predef_scores)

        predef_scores_dict = defaultdict(lambda: list(self.score_set_))
        predef_scores_dict.update({predef_features[i]: [s] for i, s in enumerate(predef_scores)})

        number_features = X.shape[1]
        remaining_features = set(range(number_features))
        self.stage_clfs = []
        self._features = []       # Initialize features list
        self._scores = []         # Initialize scores list
        self._thresholds = []     # Initialize thresholds list

        # Initial expected entropy with no features
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

        # Compute the expected entropy reduction
        delta_loss = current_loss - new_loss

        # Compute the cost of the new feature(s)
        feature_cost = np.sum(np.array(self.feature_costs)[feature_extension])
        
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