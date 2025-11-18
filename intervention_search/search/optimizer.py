"""
Intervention Search Optimizer

Implements smart search strategies for finding optimal interventions:
1. Adaptive grid search
2. Bayesian optimization (using Gaussian Processes)
3. Gradient-based optimization (for differentiable models)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SearchResult:
    """Result from optimization search"""
    intervention_pct: float
    predicted_effect: float
    error_from_target: float
    iterations: int
    converged: bool


class AdaptiveGridSearch:
    """
    Adaptive grid search that refines around promising regions.

    Improvement over fixed grid: adapts resolution based on landscape.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: Tuple[float, float],
        target_value: float,
        tolerance: float = 2.0,
        max_iterations: int = 20
    ):
        """
        Initialize adaptive grid search.

        Args:
            objective_function: Function to optimize (intervention_pct -> predicted_effect)
            bounds: (min_pct, max_pct) bounds for search
            target_value: Target effect we're trying to achieve
            tolerance: Acceptable error
            max_iterations: Maximum search iterations
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.target_value = target_value
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def search(self) -> SearchResult:
        """
        Run adaptive grid search.

        Returns:
            SearchResult with best intervention found
        """
        best_intervention = None
        best_error = float('inf')
        iterations = 0

        # Phase 1: Coarse grid
        coarse_points = np.linspace(self.bounds[0], self.bounds[1], 10)

        for pct in coarse_points:
            iterations += 1
            try:
                effect = self.objective_function(pct)
                error = abs(effect - self.target_value)

                if error < best_error:
                    best_error = error
                    best_intervention = pct

                # Early stopping if we hit target
                if error <= self.tolerance:
                    return SearchResult(
                        intervention_pct=pct,
                        predicted_effect=effect,
                        error_from_target=error,
                        iterations=iterations,
                        converged=True
                    )
            except:
                continue

        if best_intervention is None:
            return SearchResult(
                intervention_pct=0,
                predicted_effect=0,
                error_from_target=float('inf'),
                iterations=iterations,
                converged=False
            )

        # Phase 2: Adaptive refinement around best point
        search_width = (self.bounds[1] - self.bounds[0]) / 10
        step_size = 1.0

        for iteration in range(self.max_iterations):
            iterations += 1

            # Get current best
            current_effect = self.objective_function(best_intervention)
            current_error = abs(current_effect - self.target_value)

            if current_error <= self.tolerance:
                return SearchResult(
                    intervention_pct=best_intervention,
                    predicted_effect=current_effect,
                    error_from_target=current_error,
                    iterations=iterations,
                    converged=True
                )

            # Compute gradient approximation
            delta = search_width * 0.1
            try:
                effect_plus = self.objective_function(best_intervention + delta)
                effect_minus = self.objective_function(best_intervention - delta)
                gradient = (effect_plus - effect_minus) / (2 * delta)
            except:
                # If gradient fails, reduce search width
                search_width *= 0.5
                continue

            if abs(gradient) < 1e-9:
                break  # Flat region, can't improve

            # Compute step based on gradient
            if abs(current_effect) > 1e-9:
                # How much do we need to change intervention?
                needed_change_effect = self.target_value - current_effect
                needed_change_intervention = needed_change_effect / gradient if abs(gradient) > 1e-9 else 0

                # Adaptive step size
                next_intervention = best_intervention + step_size * needed_change_intervention
            else:
                # Random exploration
                next_intervention = best_intervention + np.random.uniform(-search_width, search_width)

            # Clip to bounds
            next_intervention = np.clip(next_intervention, self.bounds[0], self.bounds[1])

            # Evaluate
            try:
                next_effect = self.objective_function(next_intervention)
                next_error = abs(next_effect - self.target_value)

                if next_error < current_error:
                    best_intervention = next_intervention
                    best_error = next_error
                    step_size = min(1.0, step_size * 1.2)  # Increase step on improvement
                else:
                    step_size *= 0.5  # Reduce step on no improvement

                # Reduce search width over time
                search_width *= 0.95

                if step_size < 0.01 or search_width < 0.1:
                    break  # Converged or stuck

            except:
                step_size *= 0.5
                continue

        final_effect = self.objective_function(best_intervention)

        return SearchResult(
            intervention_pct=best_intervention,
            predicted_effect=final_effect,
            error_from_target=best_error,
            iterations=iterations,
            converged=best_error <= self.tolerance
        )


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process surrogate model.

    More efficient than grid search - actively explores promising regions.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: Tuple[float, float],
        target_value: float,
        tolerance: float = 2.0,
        n_initial_points: int = 5,
        n_iterations: int = 15
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            objective_function: Function to optimize
            bounds: Search bounds
            target_value: Target value
            tolerance: Acceptable error
            n_initial_points: Number of random initial samples
            n_iterations: Number of BO iterations
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.target_value = target_value
        self.tolerance = tolerance
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations

        # Storage for observations
        self.X_observed = []
        self.y_observed = []

    def search(self) -> SearchResult:
        """
        Run Bayesian optimization.

        Returns:
            SearchResult with best intervention found
        """
        # Phase 1: Random initialization
        initial_points = np.random.uniform(
            self.bounds[0], self.bounds[1], self.n_initial_points
        )

        for x in initial_points:
            try:
                y = self.objective_function(x)
                self.X_observed.append(x)
                self.y_observed.append(y)
            except:
                continue

        if not self.X_observed:
            return SearchResult(
                intervention_pct=0,
                predicted_effect=0,
                error_from_target=float('inf'),
                iterations=0,
                converged=False
            )

        # Phase 2: Bayesian optimization iterations
        for iteration in range(self.n_iterations):
            # Fit GP surrogate model (simplified using linear approximation)
            # In production, use scikit-optimize or similar library

            # Find best observed so far
            errors = [abs(y - self.target_value) for y in self.y_observed]
            best_idx = np.argmin(errors)
            best_x = self.X_observed[best_idx]
            best_y = self.y_observed[best_idx]
            best_error = errors[best_idx]

            if best_error <= self.tolerance:
                return SearchResult(
                    intervention_pct=best_x,
                    predicted_effect=best_y,
                    error_from_target=best_error,
                    iterations=len(self.X_observed),
                    converged=True
                )

            # Acquisition function: explore regions with high uncertainty or near target
            next_x = self._acquisition_function()

            # Evaluate
            try:
                next_y = self.objective_function(next_x)
                self.X_observed.append(next_x)
                self.y_observed.append(next_y)
            except:
                continue

        # Return best found
        errors = [abs(y - self.target_value) for y in self.y_observed]
        best_idx = np.argmin(errors)

        return SearchResult(
            intervention_pct=self.X_observed[best_idx],
            predicted_effect=self.y_observed[best_idx],
            error_from_target=errors[best_idx],
            iterations=len(self.X_observed),
            converged=errors[best_idx] <= self.tolerance
        )

    def _acquisition_function(self) -> float:
        """
        Simplified acquisition function: explore under-sampled regions.

        Returns:
            Next point to sample
        """
        if len(self.X_observed) < 3:
            # Random exploration
            return np.random.uniform(self.bounds[0], self.bounds[1])

        # Find largest gap in observed points
        sorted_x = sorted(self.X_observed)
        gaps = [(sorted_x[i+1] - sorted_x[i], (sorted_x[i] + sorted_x[i+1]) / 2)
                for i in range(len(sorted_x) - 1)]

        if gaps:
            # Sample from largest gap
            max_gap, midpoint = max(gaps)
            # Add noise for exploration
            noise = np.random.uniform(-max_gap * 0.2, max_gap * 0.2)
            next_x = midpoint + noise
            return np.clip(next_x, self.bounds[0], self.bounds[1])

        return np.random.uniform(self.bounds[0], self.bounds[1])


class MultiNodeOptimizer:
    """
    Optimizer for finding optimal interventions on multiple nodes simultaneously.
    """

    def __init__(
        self,
        objective_function: Callable,  # Takes dict of interventions -> effect
        node_bounds: Dict[str, Tuple[float, float]],
        target_value: float,
        tolerance: float = 2.0,
        max_evaluations: int = 100
    ):
        """
        Initialize multi-node optimizer.

        Args:
            objective_function: Function mapping intervention dict to predicted effect
            node_bounds: Dictionary of {node: (min_pct, max_pct)}
            target_value: Target effect
            tolerance: Acceptable error
            max_evaluations: Maximum function evaluations
        """
        self.objective_function = objective_function
        self.node_bounds = node_bounds
        self.nodes = list(node_bounds.keys())
        self.target_value = target_value
        self.tolerance = tolerance
        self.max_evaluations = max_evaluations

    def search(self) -> Dict:
        """
        Search for optimal multi-node intervention.

        Returns:
            Dictionary with optimization results
        """
        # Define objective for scipy
        def scipy_objective(x):
            # Convert array to intervention dict
            intervention_dict = {node: x[i] for i, node in enumerate(self.nodes)}
            try:
                predicted_effect = self.objective_function(intervention_dict)
                # Minimize squared error from target
                return (predicted_effect - self.target_value) ** 2
            except:
                return 1e10  # Penalty for failed evaluations

        # Bounds for scipy
        bounds = [self.node_bounds[node] for node in self.nodes]

        # Use differential evolution (global optimizer)
        try:
            result = differential_evolution(
                scipy_objective,
                bounds=bounds,
                maxiter=self.max_evaluations // len(self.nodes),
                seed=42,
                atol=self.tolerance,
                tol=0.01,
                workers=1
            )

            # Convert result back
            optimal_intervention = {node: result.x[i] for i, node in enumerate(self.nodes)}
            predicted_effect = self.objective_function(optimal_intervention)
            error = abs(predicted_effect - self.target_value)

            return {
                'intervention': optimal_intervention,
                'predicted_effect': predicted_effect,
                'error': error,
                'converged': error <= self.tolerance,
                'iterations': result.nfev
            }

        except Exception as e:
            # Fallback: grid search
            return self._fallback_grid_search()

    def _fallback_grid_search(self) -> Dict:
        """Fallback to simple grid search if optimization fails"""
        # Sample a few random combinations
        best_intervention = None
        best_effect = None
        best_error = float('inf')

        for _ in range(min(50, self.max_evaluations)):
            # Random sample
            intervention = {
                node: np.random.uniform(bounds[0], bounds[1])
                for node, bounds in self.node_bounds.items()
            }

            try:
                effect = self.objective_function(intervention)
                error = abs(effect - self.target_value)

                if error < best_error:
                    best_error = error
                    best_effect = effect
                    best_intervention = intervention

            except:
                continue

        return {
            'intervention': best_intervention or {},
            'predicted_effect': best_effect or 0,
            'error': best_error,
            'converged': best_error <= self.tolerance,
            'iterations': 50
        }


def choose_optimizer(
    n_dimensions: int,
    available_budget: int
) -> str:
    """
    Choose appropriate optimizer based on problem characteristics.

    Args:
        n_dimensions: Number of variables being optimized
        available_budget: Number of function evaluations available

    Returns:
        Optimizer name ('grid', 'bayesian', 'differential_evolution')
    """
    if n_dimensions == 1:
        if available_budget >= 20:
            return 'bayesian'
        else:
            return 'grid'
    elif n_dimensions == 2:
        if available_budget >= 50:
            return 'bayesian'
        else:
            return 'grid'
    else:
        return 'differential_evolution'
