"""
Causal Path Sensitivity Analyzer

Analyzes which causal paths contribute most to intervention effects and uncertainty.
This enables targeted, focused interventions by identifying the most reliable paths.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..utils.causal_paths import (
    enumerate_all_paths,
    CausalPath,
    compute_path_quality_score
)


@dataclass
class PathSensitivityResult:
    """Results from path sensitivity analysis"""
    path: CausalPath
    quality_metrics: Dict
    estimated_effect: float
    estimated_uncertainty: float
    contribution_weight: float  # Relative contribution to total effect


class PathSensitivityAnalyzer:
    """
    Analyzes sensitivity of intervention effects to specific causal paths.

    Key insight: When multiple paths exist from intervention → outcome,
    some paths may be reliable while others are unreliable. This analyzer
    identifies which paths to trust and focus on.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        model_metrics: Dict[str, Dict],
        edge_elasticities: Optional[Dict[Tuple[str, str], float]] = None
    ):
        """
        Initialize path sensitivity analyzer.

        Args:
            graph: Causal DAG
            model_metrics: Model quality metrics
            edge_elasticities: Optional edge elasticities for effect estimation
        """
        self.graph = graph
        self.model_metrics = model_metrics
        self.edge_elasticities = edge_elasticities or {}

    def analyze_intervention_paths(
        self,
        intervention_node: str,
        outcome_node: str,
        max_paths: int = 20
    ) -> List[PathSensitivityResult]:
        """
        Analyze all causal paths from intervention to outcome.

        Args:
            intervention_node: Node being intervened on
            outcome_node: Target outcome node
            max_paths: Maximum number of paths to analyze

        Returns:
            List of PathSensitivityResult, sorted by quality
        """
        # Enumerate paths
        all_paths = enumerate_all_paths(self.graph, intervention_node, outcome_node, max_length=6)

        if not all_paths:
            return []

        # Limit to max_paths
        paths_to_analyze = all_paths[:max_paths]

        # Analyze each path
        results = []
        for path in paths_to_analyze:
            result = self._analyze_single_path(path, outcome_node)
            if result is not None:
                results.append(result)

        # Sort by quality
        results.sort(key=lambda x: x.quality_metrics['overall_quality'], reverse=True)

        # Compute contribution weights (based on quality)
        if results:
            total_quality = sum(r.quality_metrics['overall_quality'] for r in results)
            if total_quality > 0:
                for result in results:
                    result.contribution_weight = (
                        result.quality_metrics['overall_quality'] / total_quality
                    )

        return results

    def _analyze_single_path(
        self,
        path: CausalPath,
        outcome_node: str
    ) -> Optional[PathSensitivityResult]:
        """
        Analyze a single causal path.

        Args:
            path: CausalPath object
            outcome_node: Target outcome

        Returns:
            PathSensitivityResult or None
        """
        # Compute quality metrics
        quality_metrics = compute_path_quality_score(path, self.model_metrics, self.graph)

        # Estimate effect magnitude along this path
        estimated_effect = self._estimate_path_effect(path)

        # Estimate uncertainty
        estimated_uncertainty = quality_metrics['accumulated_rmse']

        return PathSensitivityResult(
            path=path,
            quality_metrics=quality_metrics,
            estimated_effect=estimated_effect,
            estimated_uncertainty=estimated_uncertainty,
            contribution_weight=0.0  # Will be set later
        )

    def _estimate_path_effect(self, path: CausalPath) -> float:
        """
        Estimate the effect magnitude along a path using edge elasticities.

        Args:
            path: CausalPath object

        Returns:
            Estimated effect multiplier (1.0 = no change)
        """
        if not self.edge_elasticities:
            # If no elasticities, use path length as proxy
            return 0.8 ** path.length  # Exponential decay

        # Multiply elasticities along path
        cumulative_effect = 1.0
        for edge in path.edges:
            elasticity = self.edge_elasticities.get(edge, 1.0)
            cumulative_effect *= elasticity

        return cumulative_effect

    def recommend_best_path(
        self,
        intervention_node: str,
        outcome_node: str,
        min_quality_threshold: float = 0.5
    ) -> Optional[PathSensitivityResult]:
        """
        Recommend the single best causal path for intervention.

        Args:
            intervention_node: Intervention node
            outcome_node: Target outcome
            min_quality_threshold: Minimum quality to consider

        Returns:
            Best PathSensitivityResult or None
        """
        results = self.analyze_intervention_paths(intervention_node, outcome_node)

        # Filter by quality
        high_quality_results = [
            r for r in results
            if r.quality_metrics['overall_quality'] >= min_quality_threshold
        ]

        if not high_quality_results:
            return None

        return high_quality_results[0]  # Already sorted by quality

    def decompose_total_effect(
        self,
        intervention_node: str,
        outcome_node: str,
        simulated_total_effect: float
    ) -> Dict:
        """
        Decompose the total effect into contributions from individual paths.

        Args:
            intervention_node: Intervention node
            outcome_node: Outcome node
            simulated_total_effect: Total effect from simulation

        Returns:
            Dictionary with decomposition
        """
        path_results = self.analyze_intervention_paths(intervention_node, outcome_node)

        if not path_results:
            return {
                'total_effect': simulated_total_effect,
                'path_contributions': [],
                'warning': 'No paths found'
            }

        # Estimate contribution of each path
        path_contributions = []
        for result in path_results:
            # Weighted by quality and estimated effect
            contribution_estimate = (
                simulated_total_effect *
                result.contribution_weight *
                result.estimated_effect
            )

            path_contributions.append({
                'path': str(result.path),
                'quality': result.quality_metrics['overall_quality'],
                'estimated_contribution': contribution_estimate,
                'uncertainty': result.estimated_uncertainty,
                'min_r2': result.quality_metrics['min_r2'],
                'length': result.path.length
            })

        # Sort by estimated contribution
        path_contributions.sort(key=lambda x: abs(x['estimated_contribution']), reverse=True)

        # Identify dominant paths (contribute >10% each)
        dominant_paths = [p for p in path_contributions if abs(p['estimated_contribution']) > abs(simulated_total_effect) * 0.1]

        return {
            'total_effect': simulated_total_effect,
            'num_paths': len(path_results),
            'path_contributions': path_contributions,
            'dominant_paths': dominant_paths,
            'most_reliable_path': path_contributions[0] if path_contributions else None
        }

    def compare_intervention_paths(
        self,
        intervention_options: List[str],
        outcome_node: str
    ) -> Dict:
        """
        Compare path quality for multiple intervention options.

        Args:
            intervention_options: List of potential intervention nodes
            outcome_node: Target outcome

        Returns:
            Comparison dictionary
        """
        comparisons = []

        for intervention_node in intervention_options:
            best_path = self.recommend_best_path(intervention_node, outcome_node)

            if best_path is not None:
                comparisons.append({
                    'intervention_node': intervention_node,
                    'best_path': str(best_path.path),
                    'quality_score': best_path.quality_metrics['overall_quality'],
                    'min_r2': best_path.quality_metrics['min_r2'],
                    'path_length': best_path.path.length,
                    'uncertainty': best_path.estimated_uncertainty
                })

        # Sort by quality
        comparisons.sort(key=lambda x: x['quality_score'], reverse=True)

        return {
            'comparisons': comparisons,
            'best_intervention': comparisons[0] if comparisons else None,
            'recommendation': self._generate_path_recommendation(comparisons)
        }

    def _generate_path_recommendation(self, comparisons: List[Dict]) -> str:
        """Generate human-readable recommendation"""
        if not comparisons:
            return "No reliable paths found for any intervention option."

        best = comparisons[0]

        if best['quality_score'] >= 0.8:
            confidence = "high confidence"
        elif best['quality_score'] >= 0.6:
            confidence = "moderate confidence"
        else:
            confidence = "low confidence"

        return (
            f"Recommend intervening on '{best['intervention_node']}' "
            f"via path: {best['best_path']} "
            f"(quality: {best['quality_score']:.2f}, {confidence})"
        )

    def identify_uncertainty_hotspots(
        self,
        intervention_node: str,
        outcome_node: str
    ) -> Dict:
        """
        Identify which nodes in causal paths contribute most to uncertainty.

        Args:
            intervention_node: Intervention node
            outcome_node: Outcome node

        Returns:
            Dictionary identifying uncertainty hotspots
        """
        path_results = self.analyze_intervention_paths(intervention_node, outcome_node)

        if not path_results:
            return {'hotspots': [], 'warning': 'No paths found'}

        # Count node appearances and accumulate uncertainty
        node_uncertainty = {}
        node_appearances = {}

        for result in path_results:
            for i, node in enumerate(result.path.nodes[1:], 1):  # Skip intervention node
                if node not in node_uncertainty:
                    node_uncertainty[node] = []
                    node_appearances[node] = 0

                node_appearances[node] += 1

                # Get RMSE for this node
                if node in self.model_metrics:
                    metrics = self.model_metrics[node]
                    if metrics.get('model_type') == 'regression':
                        rmse = metrics.get('rmse', 0)
                        node_uncertainty[node].append(rmse)

        # Compute average uncertainty per node
        hotspots = []
        for node, rmse_list in node_uncertainty.items():
            if rmse_list:
                avg_rmse = np.mean(rmse_list)
                appearances = node_appearances[node]

                # Hotspot score: high RMSE × many appearances
                hotspot_score = avg_rmse * np.log1p(appearances)

                hotspots.append({
                    'node': node,
                    'avg_rmse': avg_rmse,
                    'appearances': appearances,
                    'hotspot_score': hotspot_score,
                    'r2': self.model_metrics.get(node, {}).get('r2_score', 0)
                })

        # Sort by hotspot score
        hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)

        return {
            'hotspots': hotspots,
            'top_contributor': hotspots[0] if hotspots else None,
            'recommendation': (
                f"Improving model quality for '{hotspots[0]['node']}' would have biggest impact on uncertainty"
                if hotspots else "No uncertainty hotspots identified"
            )
        }
