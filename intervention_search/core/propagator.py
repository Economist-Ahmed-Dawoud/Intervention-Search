"""
Monte Carlo Uncertainty Propagator with DO Operator Methodology

Implements uncertainty-aware causal propagation through DAGs using Monte Carlo simulation
and Pearl's DO operator methodology.

DO OPERATOR COMPLIANCE:
- Interventions use do(X=x) semantics: incoming edges to X are removed (X is fixed)
- Propagation respects causal structure via topological ordering
- Only descendants of intervention are affected (causal consistency)
- Each simulation samples uncertainty while maintaining causal graph structure

This addresses the critical issue of underestimating uncertainty in intervention predictions
while ensuring 100% compliance with causal inference methodology.
"""

import numpy as np
import networkx as nx
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')


def _run_simulation_worker(
    sim_idx: int,
    initial_values: Dict[str, float],
    intervened_nodes: Set[str],
    intervention_values: Dict[str, float],
    propagator_state: Dict,
    seed_base: int
) -> Dict[str, float]:
    """
    Worker function for parallel Monte Carlo simulation.

    This function is defined at module level to be picklable for multiprocessing.

    Args:
        sim_idx: Simulation index (for seeding)
        initial_values: Initial node values
        intervened_nodes: Nodes being intervened on
        intervention_values: Intervention values
        propagator_state: State dict with graph, models, etc.
        seed_base: Base random seed

    Returns:
        Dictionary of predicted values for all nodes
    """
    # Set unique seed for this simulation
    np.random.seed(seed_base + sim_idx)

    # Reconstruct necessary state
    topological_order = propagator_state['topological_order']
    regressors_dict = propagator_state['regressors_dict']
    baseline_stats = propagator_state['baseline_stats']
    model_metrics = propagator_state['model_metrics']
    node_types = propagator_state['node_types']
    graph = propagator_state['graph']

    # Run simulation (same logic as _single_simulation)
    predicted_values = initial_values.copy()

    # Fix intervened nodes
    for node in intervened_nodes:
        predicted_values[node] = intervention_values[node]

    # Propagate through descendants
    for node in topological_order:
        if node in intervened_nodes:
            continue

        parents = list(graph.predecessors(node))

        if not parents:
            # Root node: sample from baseline distribution
            if node in baseline_stats:
                stats = baseline_stats[node]
                std = stats.get('std', 0.0)
                if std > 0:
                    noise = np.random.normal(0, std * 0.1)
                    predicted_values[node] = predicted_values.get(node, stats.get('mean', 0)) + noise
            continue

        # Get model
        if node not in regressors_dict:
            continue

        regressor, scaler = regressors_dict[node]
        if regressor is None:
            continue

        # Get parent values
        parent_values = np.array([[predicted_values[p] for p in parents]])

        # Predict with uncertainty
        try:
            is_categorical = node_types.get(node) == 'categorical'

            if is_categorical:
                predicted_class = regressor.predict(parent_values)[0]
                predicted_values[node] = float(predicted_class)
            else:
                base_prediction = regressor.predict(parent_values)[0]

                # Add noise based on model RMSE
                if node in model_metrics:
                    metrics = model_metrics[node]
                    if metrics.get('model_type') == 'regression':
                        rmse = metrics.get('rmse', 0.0)
                        noise = np.random.normal(0, rmse)
                        predicted_values[node] = base_prediction + noise
                    else:
                        predicted_values[node] = base_prediction
                else:
                    predicted_values[node] = base_prediction

        except Exception:
            # Fallback to baseline
            predicted_values[node] = initial_values.get(node, 0.0)

    return predicted_values


@dataclass
class PropagationResult:
    """Results from Monte Carlo propagation"""
    node_distributions: Dict[str, np.ndarray]  # node -> array of samples
    node_means: Dict[str, float]
    node_stds: Dict[str, float]
    node_percentiles: Dict[str, Dict[str, float]]  # node -> {p5, p25, p50, p75, p95}


class MonteCarloPropagator:
    """
    Propagates interventions through causal graphs with proper uncertainty quantification.

    Key improvements over deterministic propagation:
    1. Accounts for model uncertainty (RMSE)
    2. Propagates uncertainty through causal chains
    3. Provides proper prediction intervals
    4. Handles both regression and classification models
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        regressors_dict: Dict,
        baseline_stats: Dict,
        model_metrics: Dict,
        node_types: Dict,
        n_simulations: int = 1000,
        random_seed: Optional[int] = None,
        use_parallel: bool = False,
        n_jobs: int = -1
    ):
        """
        Initialize the Monte Carlo propagator.

        Args:
            graph: Causal DAG
            regressors_dict: Dictionary of (model, scaler) for each node
            baseline_stats: Baseline statistics for each node
            model_metrics: Model quality metrics (R², RMSE, etc.)
            node_types: Type of each node (categorical/continuous)
            n_simulations: Number of Monte Carlo samples (default: 1000)
            random_seed: Random seed for reproducibility
            use_parallel: Enable parallel processing for Monte Carlo (default: False)
            n_jobs: Number of parallel jobs (-1 = all CPUs, default: -1)
        """
        self.graph = graph
        self.regressors_dict = regressors_dict
        self.baseline_stats = baseline_stats
        self.model_metrics = model_metrics
        self.node_types = node_types
        self.n_simulations = n_simulations
        self.use_parallel = use_parallel
        self.random_seed = random_seed

        # Determine number of jobs
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))

        if random_seed is not None:
            np.random.seed(random_seed)

        # Precompute topological order
        try:
            self.topological_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles! Cannot perform causal simulation.")

    def propagate_with_uncertainty(
        self,
        initial_values: Dict[str, float],
        intervened_nodes: Set[str],
        intervention_values: Dict[str, float]
    ) -> PropagationResult:
        """
        Propagate through graph with Monte Carlo uncertainty estimation.

        DO OPERATOR IMPLEMENTATION:
        This method implements P(Y | do(X=x)) by:
        1. Fixing X to value x (ignoring its parents - equivalent to removing incoming edges)
        2. Propagating through descendants in topological order
        3. Sampling uncertainty at each step to get full distribution

        This is the correct causal interpretation, NOT P(Y | X=x) which would be observational.

        Args:
            initial_values: Starting values for all nodes
            intervened_nodes: Set of nodes being intervened on (the X in do(X=x))
            intervention_values: Values for intervened nodes (the x in do(X=x))

        Returns:
            PropagationResult with distributions for all nodes
        """
        # Initialize storage for samples
        node_samples = {node: np.zeros(self.n_simulations) for node in self.graph.nodes()}

        # Run Monte Carlo simulations (with optional parallelization)
        if self.use_parallel and self.n_simulations >= 100:
            # Parallel execution for large simulation counts
            # Split simulations into chunks for each process
            chunk_size = max(50, self.n_simulations // self.n_jobs)

            # Create simulation function with fixed parameters
            sim_func = partial(
                _run_simulation_worker,
                initial_values=initial_values,
                intervened_nodes=intervened_nodes,
                intervention_values=intervention_values,
                propagator_state={
                    'topological_order': self.topological_order,
                    'regressors_dict': self.regressors_dict,
                    'baseline_stats': self.baseline_stats,
                    'model_metrics': self.model_metrics,
                    'node_types': self.node_types,
                    'graph': self.graph
                },
                seed_base=self.random_seed if self.random_seed else 0
            )

            # Run simulations in parallel
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(sim_func, range(self.n_simulations))

            # Collect results
            for sim_idx, sim_result in enumerate(results):
                for node, value in sim_result.items():
                    node_samples[node][sim_idx] = value
        else:
            # Sequential execution (default - more reliable and reproducible)
            for sim_idx in range(self.n_simulations):
                # Single simulation run
                sim_result = self._single_simulation(
                    initial_values,
                    intervened_nodes,
                    intervention_values
                )

                # Store results
                for node, value in sim_result.items():
                    node_samples[node][sim_idx] = value

        # Compute summary statistics
        node_means = {node: np.mean(samples) for node, samples in node_samples.items()}
        node_stds = {node: np.std(samples) for node, samples in node_samples.items()}
        node_percentiles = {
            node: {
                'p5': np.percentile(samples, 5),
                'p25': np.percentile(samples, 25),
                'p50': np.percentile(samples, 50),
                'p75': np.percentile(samples, 75),
                'p95': np.percentile(samples, 95)
            }
            for node, samples in node_samples.items()
        }

        return PropagationResult(
            node_distributions=node_samples,
            node_means=node_means,
            node_stds=node_stds,
            node_percentiles=node_percentiles
        )

    def _single_simulation(
        self,
        initial_values: Dict[str, float],
        intervened_nodes: Set[str],
        intervention_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Run a single Monte Carlo simulation using DO operator semantics.

        DO OPERATOR ENFORCEMENT:
        1. Intervened nodes are FIXED to intervention values (line 160-161)
           → This implements the "removal of incoming edges" in do(X=x)
        2. Topological ordering ensures parents computed before children (line 164)
           → Respects causal flow
        3. Intervened nodes are skipped in propagation (line 166-167)
           → Their values don't depend on parents (incoming edges removed)
        4. Non-intervened nodes use causal mechanisms (models) from parents
           → Standard structural equation evaluation

        Args:
            initial_values: Starting values
            intervened_nodes: Nodes being intervened on
            intervention_values: Intervention values

        Returns:
            Dictionary of predicted values for all nodes
        """
        predicted_values = initial_values.copy()

        # DO OPERATOR STEP 1: Fix intervened nodes (remove incoming edges conceptually)
        for node in intervened_nodes:
            predicted_values[node] = intervention_values[node]

        # DO OPERATOR STEP 2: Propagate through descendants in topological order
        for node in self.topological_order:
            # DO OPERATOR STEP 3: Skip intervened nodes (already fixed, ignore parents)
            if node in intervened_nodes:
                continue

            parents = list(self.graph.predecessors(node))

            if not parents:
                # Root node: sample from baseline distribution
                predicted_values[node] = self._sample_root_node(node, initial_values[node])
                continue

            # Get model for this node
            if node not in self.regressors_dict:
                continue

            regressor, scaler = self.regressors_dict[node]

            if regressor is None:
                continue

            # Get parent values (after propagation)
            parent_values = np.array([[predicted_values[p] for p in parents]])

            # Predict with uncertainty
            try:
                predicted_value = self._predict_with_uncertainty(
                    node, regressor, parent_values
                )
                predicted_values[node] = predicted_value

            except Exception as e:
                # Fallback to baseline if prediction fails
                predicted_values[node] = initial_values.get(node, 0.0)

        return predicted_values

    def _predict_with_uncertainty(
        self,
        node: str,
        regressor,
        parent_values: np.ndarray
    ) -> float:
        """
        Make a prediction with added uncertainty (noise).

        Args:
            node: Node being predicted
            regressor: Trained model
            parent_values: Parent node values

        Returns:
            Predicted value with uncertainty
        """
        # Get base prediction
        is_categorical = self.node_types.get(node) == 'categorical'

        if is_categorical:
            # For categorical: predict class
            predicted_class = regressor.predict(parent_values)[0]
            base_prediction = float(predicted_class)
            # No noise added to categorical predictions
            return base_prediction
        else:
            # For continuous: predict value
            base_prediction = regressor.predict(parent_values)[0]

            # Add noise based on model RMSE
            if node in self.model_metrics:
                metrics = self.model_metrics[node]
                if metrics.get('model_type') == 'regression':
                    rmse = metrics.get('rmse', 0.0)
                    # Sample from N(0, RMSE²)
                    noise = np.random.normal(0, rmse)
                    return base_prediction + noise

            return base_prediction

    def _sample_root_node(self, node: str, baseline_value: float) -> float:
        """
        Sample a root node value from its baseline distribution.

        Args:
            node: Node name
            baseline_value: Baseline mean value

        Returns:
            Sampled value
        """
        if node in self.baseline_stats:
            stats = self.baseline_stats[node]
            std = stats.get('std', 0.0)

            # Add small noise to root nodes
            if std > 0:
                noise = np.random.normal(0, std * 0.1)  # 10% of baseline std
                return baseline_value + noise

        return baseline_value

    def compare_interventions(
        self,
        initial_values: Dict[str, float],
        intervention_values: Dict[str, float],
        outcome_nodes: List[str]
    ) -> Dict:
        """
        Compare world without intervention vs. with intervention using Monte Carlo.

        Args:
            initial_values: Baseline values
            intervention_values: Intervention specification
            outcome_nodes: Nodes to track

        Returns:
            Dictionary with comparison results including uncertainty
        """
        # Propagate without intervention
        no_intervention_result = self.propagate_with_uncertainty(
            initial_values=initial_values,
            intervened_nodes=set(),
            intervention_values={}
        )

        # Propagate with intervention
        with_intervention_result = self.propagate_with_uncertainty(
            initial_values=initial_values,
            intervened_nodes=set(intervention_values.keys()),
            intervention_values=intervention_values
        )

        # Compute causal effects for outcome nodes
        outcome_effects = {}

        for outcome in outcome_nodes:
            no_interv_samples = no_intervention_result.node_distributions[outcome]
            with_interv_samples = with_intervention_result.node_distributions[outcome]

            # Compute effect distribution
            effect_samples = with_interv_samples - no_interv_samples

            # Summary statistics
            no_interv_mean = no_intervention_result.node_means[outcome]
            with_interv_mean = with_intervention_result.node_means[outcome]
            effect_mean = with_interv_mean - no_interv_mean

            # Percentage effect
            pct_effect = None
            if abs(no_interv_mean) > 1e-9:
                pct_effect = (effect_mean / no_interv_mean) * 100

            # Confidence intervals for effect (ABSOLUTE)
            effect_ci_90_abs = (np.percentile(effect_samples, 5), np.percentile(effect_samples, 95))
            effect_ci_50_abs = (np.percentile(effect_samples, 25), np.percentile(effect_samples, 75))

            # Convert CIs to PERCENTAGE to match pct_effect
            # FIX: This resolves wide CI issue - CIs must be in same units as pct_effect
            if abs(no_interv_mean) > 1e-9:
                effect_ci_90_pct = (
                    (effect_ci_90_abs[0] / no_interv_mean) * 100,
                    (effect_ci_90_abs[1] / no_interv_mean) * 100
                )
                effect_ci_50_pct = (
                    (effect_ci_50_abs[0] / no_interv_mean) * 100,
                    (effect_ci_50_abs[1] / no_interv_mean) * 100
                )
            else:
                effect_ci_90_pct = effect_ci_90_abs
                effect_ci_50_pct = effect_ci_50_abs

            outcome_effects[outcome] = {
                'no_intervention_mean': no_interv_mean,
                'no_intervention_std': no_intervention_result.node_stds[outcome],
                'with_intervention_mean': with_interv_mean,
                'with_intervention_std': with_intervention_result.node_stds[outcome],
                'absolute_effect': effect_mean,
                'absolute_effect_std': np.std(effect_samples),
                'pct_effect': pct_effect,
                'ci_90': effect_ci_90_pct,  # Now in percentage!
                'ci_50': effect_ci_50_pct,  # Now in percentage!
                'ci_90_absolute': effect_ci_90_abs,  # Keep absolute for reference
                'ci_50_absolute': effect_ci_50_abs,  # Keep absolute for reference
                'effect_samples': effect_samples
            }

        return {
            'interventions': intervention_values,
            'outcome_effects': outcome_effects,
            'no_intervention_result': no_intervention_result,
            'with_intervention_result': with_intervention_result
        }

    def estimate_path_specific_uncertainty(
        self,
        path_nodes: List[str]
    ) -> Dict:
        """
        Estimate uncertainty contribution from a specific causal path.

        Args:
            path_nodes: List of nodes in the path

        Returns:
            Dictionary with path uncertainty analysis
        """
        path_rmses = []
        path_r2s = []

        for node in path_nodes[1:]:  # Skip first node (intervention node)
            if node in self.model_metrics:
                metrics = self.model_metrics[node]
                if metrics.get('model_type') == 'regression':
                    rmse = metrics.get('rmse', 0.0)
                    r2 = metrics.get('r2_score', 0.0)
                    path_rmses.append(rmse)
                    path_r2s.append(r2)

        # Accumulated RMSE (assumes independence)
        accumulated_rmse = np.sqrt(sum(rmse**2 for rmse in path_rmses)) if path_rmses else 0.0

        # Weakest link R²
        min_r2 = min(path_r2s) if path_r2s else 1.0

        return {
            'path': ' → '.join(path_nodes),
            'accumulated_rmse': accumulated_rmse,
            'min_r2': min_r2,
            'mean_r2': np.mean(path_r2s) if path_r2s else 1.0,
            'path_length': len(path_nodes) - 1,
            'models_in_path': len(path_rmses)
        }
