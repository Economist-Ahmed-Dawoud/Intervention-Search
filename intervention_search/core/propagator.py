"""
Monte Carlo Uncertainty Propagator

Implements uncertainty-aware causal propagation through DAGs using Monte Carlo simulation.
This addresses the critical issue of underestimating uncertainty in intervention predictions.
"""

import numpy as np
import networkx as nx
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


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
        random_seed: Optional[int] = None
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
        """
        self.graph = graph
        self.regressors_dict = regressors_dict
        self.baseline_stats = baseline_stats
        self.model_metrics = model_metrics
        self.node_types = node_types
        self.n_simulations = n_simulations

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

        Args:
            initial_values: Starting values for all nodes
            intervened_nodes: Set of nodes being intervened on
            intervention_values: Values for intervened nodes

        Returns:
            PropagationResult with distributions for all nodes
        """
        # Initialize storage for samples
        node_samples = {node: np.zeros(self.n_simulations) for node in self.graph.nodes()}

        # Run Monte Carlo simulations
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
        Run a single Monte Carlo simulation.

        Args:
            initial_values: Starting values
            intervened_nodes: Nodes being intervened on
            intervention_values: Intervention values

        Returns:
            Dictionary of predicted values for all nodes
        """
        predicted_values = initial_values.copy()

        # Set intervention values (these are fixed across simulations)
        for node in intervened_nodes:
            predicted_values[node] = intervention_values[node]

        # Process each node in topological order
        for node in self.topological_order:
            # Skip intervened nodes
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

            # Confidence intervals for effect
            effect_ci_90 = (np.percentile(effect_samples, 5), np.percentile(effect_samples, 95))
            effect_ci_50 = (np.percentile(effect_samples, 25), np.percentile(effect_samples, 75))

            outcome_effects[outcome] = {
                'no_intervention_mean': no_interv_mean,
                'no_intervention_std': no_intervention_result.node_stds[outcome],
                'with_intervention_mean': with_interv_mean,
                'with_intervention_std': with_intervention_result.node_stds[outcome],
                'absolute_effect': effect_mean,
                'absolute_effect_std': np.std(effect_samples),
                'pct_effect': pct_effect,
                'ci_90': effect_ci_90,
                'ci_50': effect_ci_50,
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
