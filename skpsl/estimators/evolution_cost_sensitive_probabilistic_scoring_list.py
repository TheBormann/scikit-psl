import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from functools import partial

from .probabilistic_scoring_list import ProbabilisticScoringList
from .probabilistic_scoring_system import ProbabilisticScoringSystem


class EvolutionCostSensitiveProbabilisticScoringList(ProbabilisticScoringList):
    """
    Evolutionary algorithm-based cost-sensitive probabilistic scoring list classifier.
    Implements feature selection using NSGA-II via DEAP for multi-objective optimization.
    """

    def __init__(
        self,
        score_set: set,
        method="bisect",
        population_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        n_jobs: Optional[int] = None,
        stage_clf_params: Optional[dict] = None,
        selection_method = 'pareto'
    ):
        """
        Initializes the EvolutionCostSensitiveProbabilisticScoringList.

        :param score_set: Set of score values to be considered (feature weights).
        :param method: Optimization method for threshold optimization.
        :param population_size: Number of individuals in the population.
        :param generations: Number of generations to evolve.
        :param crossover_prob: Probability of performing crossover.
        :param mutation_prob: Probability of performing mutation.
        :param n_jobs: Number of parallel jobs for computation.
        :param stage_clf_params: Additional parameters for the stage classifiers.
        """
        super().__init__(
            score_set=score_set,
            method=method,
            loss_cutoff=None,  # Not used in evolutionary approach
            lookahead=None,     # Not used in evolutionary approach
            n_jobs=n_jobs,
            stage_loss=None,    # Not used directly
            cascade_loss=None,  # Not used directly
            stage_clf_params=stage_clf_params,
        )
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_costs: np.ndarray,
        predef_features: Optional[np.ndarray] = None,
        predef_scores: Optional[np.ndarray] = None,
        strict: bool = True,
    ) -> "EvolutionCostSensitiveProbabilisticScoringList":
        """
        Fits a cost-sensitive probabilistic scoring list to the given data using an evolutionary algorithm.

        :param X: Feature matrix.
        :param y: Target vector.
        :param feature_costs: Array containing the cost for each feature.
        :param predef_features: Predefined features to include.
        :param predef_scores: Predefined scores corresponding to the predefined features.
        :param strict: Whether to strictly use the predefined features.
        :return: The fitted classifier.
        """
        # Validate feature costs
        if feature_costs is None:
            raise ValueError("Feature costs must be provided.")
        if len(feature_costs) != X.shape[1]:
            raise ValueError("Length of feature_costs must equal number of features.")
        self.feature_costs = np.array(feature_costs)

        self.classes_ = np.unique(y)
        if predef_scores is not None and predef_features is not None:
            if len(predef_features) != len(predef_scores):
                raise ValueError("Length of predef_features must match length of predef_scores.")

        num_features = X.shape[1]

        # Store parameters for later use
        self._predef_features = predef_features
        self._predef_scores = predef_scores
        self._strict = strict

        # Setup DEAP framework
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize loss and cost
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Attribute generator: each feature is either selected (1) or not (0)
        toolbox.register("attr_bool", np.random.randint, 0, 2)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        toolbox.register(
            "evaluate",
            partial(
                eval_individual,
                X=X,
                y=y,
                feature_costs=self.feature_costs,
                predef_features=predef_features,
                predef_scores=predef_scores,
                strict=strict,
                score_set=self.score_set_,
                stage_clf_params=self.stage_clf_params_,
            ),
        )
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)

        # Parallel evaluation
        if self.n_jobs and self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)

        # Initialize population
        population = toolbox.population(n=self.population_size)

        # Statistics (optional)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg_loss", lambda fits: np.mean([f[0] for f in fits]))
        stats.register("avg_cost", lambda fits: np.mean([f[1] for f in fits]))
        stats.register("min_loss", lambda fits: np.min([f[0] for f in fits]))
        stats.register("min_cost", lambda fits: np.min([f[1] for f in fits]))

        # Evolutionary algorithm
        try:
            population, logbook = algorithms.eaMuPlusLambda(
                population,
                toolbox,
                mu=self.population_size,
                lambda_=self.population_size,
                cxpb=self.crossover_prob,
                mutpb=self.mutation_prob,
                ngen=self.generations,
                stats=stats,
                halloffame=None,
                verbose=True
            )
        except ValueError as ve:
            if 'a' in str(ve) and "cannot be empty" in str(ve):
                raise ValueError(f"Encountered ValueError during evolutionary algorithm: {ve}. Ensure 'score_set' is not empty.")
            else:
                raise ve

        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]

        # Convert Pareto front to solutions
        self.pareto_front = []
        for ind in pareto_front:
            selected_features = [i for i, bit in enumerate(ind) if bit == 1]
            if self._strict and self._predef_features is not None:
                selected_features = list(set(selected_features) | set(self._predef_features))

            # Assign scores
            selected_scores = []
            for feat in selected_features:
                if (self._predef_scores is not None and 
                    self._predef_features is not None and 
                    feat in self._predef_features):
                    idx = list(self._predef_features).index(feat)
                    selected_scores.append(self._predef_scores[idx])
                else:
                    if len(self.score_set_) == 0:
                        raise ValueError("Score set is empty during Pareto front processing.")
                    selected_scores.append(np.random.choice(list(self.score_set_)))

            # Fit classifier
            clf = ProbabilisticScoringSystem(
                features=selected_features,
                scores=selected_scores,
                initial_feature_thresholds=[None] * len(selected_features),
                **self.stage_clf_params_,
            ).fit(X, y)
            loss = clf.score(X, y)
            cost = self.feature_costs[selected_features].sum()

            solution = {
                'features': selected_features,
                'scores': selected_scores,
                'thresholds': clf.feature_thresholds,
                'loss': loss,
                'cost': cost,
                'classifier': clf
            }
            self.pareto_front.append(solution)

        # Cleanup DEAP creators to avoid conflicts in future runs
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        if self.n_jobs and self.n_jobs > 1:
            pool.close()
            
        return self

def eval_individual(
    individual,
    X,
    y,
    feature_costs,
    predef_features,
    predef_scores,
    strict,
    score_set,
    stage_clf_params,
):
    """
    Evaluates an individual in the evolutionary algorithm.

    :param individual: The individual to evaluate.
    :param X: Feature matrix.
    :param y: Target vector.
    :param feature_costs: Costs associated with each feature.
    :param predef_features: Predefined features to include.
    :param predef_scores: Predefined scores corresponding to the predefined features.
    :param strict: Whether to strictly use the predefined features.
    :param score_set: Set of possible scores.
    :param stage_clf_params: Parameters for the stage classifiers.
    :return: A tuple containing the loss and cost.
    """
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if strict and predef_features is not None:
        selected_features = list(set(selected_features) | set(predef_features))

    # Assign scores
    selected_scores = []
    for feat in selected_features:
        if predef_scores is not None and predef_features is not None and feat in predef_features:
            idx = list(predef_features).index(feat)
            selected_scores.append(predef_scores[idx])
        else:
            if len(score_set) == 0:
                raise ValueError("Score set is empty during evaluation.")
            selected_scores.append(np.random.choice(list(score_set)))

    # Fit classifier
    clf = ProbabilisticScoringSystem(
        features=selected_features,
        scores=selected_scores,
        initial_feature_thresholds=[None] * len(selected_features),
        **stage_clf_params,
    ).fit(X, y)
    loss = clf.score(X, y)
    cost = feature_costs[selected_features].sum()

    return loss, cost