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
    Now we store the selected features, chosen scores, cost, loss, 
    and even the fitted classifier so we don't have to refit later.
    """
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if strict and predef_features is not None:
        selected_features = list(set(selected_features) | set(predef_features))

    # Assign scores exactly once
    selected_scores = []
    for feat in selected_features:
        if (predef_scores is not None 
            and predef_features is not None 
            and feat in predef_features):
            idx = list(predef_features).index(feat)
            selected_scores.append(predef_scores[idx])
        else:
            if len(score_set) == 0:
                raise ValueError("Score set is empty during evaluation.")
            selected_scores.append(np.random.choice(list(score_set)))

    # Fit classifier exactly once
    clf = ProbabilisticScoringSystem(
        features=selected_features,
        scores=selected_scores,
        initial_feature_thresholds=[None] * len(selected_features),
        **stage_clf_params,
    ).fit(X, y)
    loss = clf.score(X, y)
    cost = feature_costs[selected_features].sum()

    # Store everything in the individual so we can retrieve it later
    individual.selected_features_ = selected_features
    individual.selected_scores_ = selected_scores
    individual.loss_ = loss
    individual.cost_ = cost
    individual.classifier_ = clf  # optional but handy

    # Return the fitness
    return loss, cost


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
        selection_method='pareto'
    ):
        """
        Initializes the EvolutionCostSensitiveProbabilisticScoringList.
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
        Fits a cost-sensitive probabilistic scoring list using NSGA-II.
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

        self._predef_features = predef_features
        self._predef_scores = predef_scores
        self._strict = strict

        # Setup DEAP
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # IMPORTANT: We use the improved eval_individual that stores results
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

        # Parallel if needed
        if self.n_jobs and self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)

        # Initialize population
        population = toolbox.population(n=self.population_size)

        # Initialize Pareto Front Hall of Fame
        pareto_front_hof = tools.ParetoFront()

        # Statistics (optional)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg_loss", lambda fits: np.mean([f[0] for f in fits]))
        stats.register("avg_cost", lambda fits: np.mean([f[1] for f in fits]))
        stats.register("min_loss", lambda fits: np.min([f[0] for f in fits]))
        stats.register("min_cost", lambda fits: np.min([f[1] for f in fits]))

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
                halloffame=pareto_front_hof,
                verbose=True
            )
        except ValueError as ve:
            if 'a' in str(ve) and "cannot be empty" in str(ve):
                raise ValueError(
                    f"Encountered ValueError during evolutionary algorithm: {ve}. "
                    "Ensure 'score_set' is not empty."
                )
            else:
                raise ve

        # Extract Pareto front
        if self.selection_method == 'pareto':
            self.pareto_front = self._extract_pareto_front_hof(pareto_front_hof)
        elif self.selection_method == 'custom':
            solutions = self._convert_population_to_solutions(population)
            self.pareto_front = self._compute_pareto_front(solutions)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}. Choose 'pareto' or 'custom'.")

        # Cleanup DEAP
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        if self.n_jobs and self.n_jobs > 1:
            pool.close()

        return self

    def _extract_pareto_front_hof(self, pareto_front_hof):
        """
        Converts the DEAP ParetoFront Hall of Fame into the internal Pareto front representation.
        Now we do NOT re-fit or re-randomize. 
        We simply read what's already stored on the individuals.
        """
        pareto_front = []
        for ind in pareto_front_hof:
            # Reuse from the individual's stored properties:
            selected_features = ind.selected_features_
            selected_scores = ind.selected_scores_
            loss = ind.loss_
            cost = ind.cost_
            clf = ind.classifier_

            # Build the solution dictionary
            solution = {
                'features': selected_features,
                'scores': selected_scores,
                'thresholds': clf.feature_thresholds,
                'loss': loss,
                'cost': cost,
                'classifier': clf
            }
            pareto_front.append(solution)
        return pareto_front

    def _convert_population_to_solutions(self, population):
        """
        Converts a DEAP population into a list of solution dictionaries
        using the stored values on each individual.
        """
        solutions = []
        for ind in population:
            selected_features = ind.selected_features_
            selected_scores = ind.selected_scores_
            loss = ind.loss_
            cost = ind.cost_
            clf = ind.classifier_

            solution = {
                'features': selected_features,
                'scores': selected_scores,
                'thresholds': clf.feature_thresholds,
                'loss': loss,
                'cost': cost,
                'classifier': clf
            }
            solutions.append(solution)
        return solutions

    def _compute_pareto_front(self, solutions):
        """
        Non-dominated sorting to compute Pareto front from solution dicts.
        """
        pareto_front = []
        for s in solutions:
            dominated = False
            for other_s in solutions:
                if other_s == s:
                    continue
                # If other_s is strictly better (or equal) in both objectives 
                # and strictly better in at least one, s is dominated
                if (other_s['loss'] <= s['loss'] and other_s['cost'] <= s['cost']) and \
                   (other_s['loss'] < s['loss'] or other_s['cost'] < s['cost']):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(s)
        return pareto_front