"""
Intervention Search - Main Public API

Simple, production-ready interface for finding optimal causal interventions.

Usage:
    searcher = InterventionSearch(causal_graph, trained_ht_model)
    results = searcher.find_interventions(
        target_outcome='sales',
        target_change=+10,
        confidence_level=0.90
    )
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

from ..core.propagator import MonteCarloPropagator
from ..core.quality_metrics import QualityGate, generate_quality_summary
from ..core.path_analyzer import PathSensitivityAnalyzer
from ..search.optimizer import AdaptiveGridSearch, BayesianOptimizer, MultiNodeOptimizer
from ..search.validators import InterventionValidator
from ..search.ranker import InterventionRanker, RankingWeights

warnings.filterwarnings('ignore')


def _generate_intervention_text_fields(
    intervention_node: str,
    outcome_node: str,
    intervention_pct_change: float,
    outcome_effect: float,
    confidence_score: float,
    quality_grade: str = 'C'
) -> dict:
    """
    Generate text templates and additional fields for platform compatibility.

    Smart template generation that recognizes positive/negative directions.

    Args:
        intervention_node: Name of the intervened variable
        outcome_node: Name of the outcome variable
        intervention_pct_change: Percentage change in intervention (e.g., -5.36)
        outcome_effect: Predicted percentage effect on outcome (e.g., -27.95)
        confidence_score: Confidence score 0-1
        quality_grade: Model quality grade (A, B, C, D, F)

    Returns:
        Dictionary with: details, summary, priority, validity, confidence,
                        opportunityValue, recommendationDetail
    """
    # Determine direction words for intervention
    if intervention_pct_change < 0:
        intervention_action = "decrease"
        intervention_action_summary = "Reduce"
    else:
        intervention_action = "increase"
        intervention_action_summary = "Increase"

    # Determine direction words for outcome
    if outcome_effect < 0:
        outcome_direction = "bring down"
        outcome_action_summary = "reduce"
    else:
        outcome_direction = "bring up"
        outcome_action_summary = "increase"

    # Format percentages for display
    intervention_pct_display = abs(intervention_pct_change)
    outcome_pct_display = round(outcome_effect, 2)

    # Build path string
    path_string = f"{intervention_node} -> {outcome_node}"

    # Build summary: "Reduce X by Y% to achieve reduce in Z"
    summary = (
        f"{intervention_action_summary} {intervention_node} by {intervention_pct_display:.1f}% "
        f"to achieve {outcome_action_summary} in {outcome_node}"
    )

    # Build recommendation detail
    recommendation_detail = (
        f"To {outcome_direction} {outcome_node} by {outcome_pct_display}%, "
        f"the easiest lever (by % change) is to {intervention_action} "
        f"{intervention_node} by {abs(intervention_pct_change):.2f}%"
    )

    # Compute priority based on confidence
    if confidence_score >= 0.7:
        priority = "High"
    elif confidence_score >= 0.5:
        priority = "Medium"
    else:
        priority = "Low"

    # Compute validity (days) based on model quality grade
    validity_map = {'A': 30, 'B': 28, 'C': 21, 'D': 14, 'F': 7}
    validity = validity_map.get(quality_grade, 21)

    # Compute confidence text
    if confidence_score >= 0.7:
        confidence_text = "High"
    elif confidence_score >= 0.5:
        confidence_text = "Medium"
    else:
        confidence_text = "Low"

    return {
        'details': [{'path': path_string}],
        'summary': summary,
        'priority': priority,
        'validity': validity,
        'confidence': confidence_text,
        'opportunityValue': None,
        'recommendationDetail': recommendation_detail
    }


def _generate_combination_text_fields(
    intervention_nodes: list,
    outcome_node: str,
    intervention_pct_changes: dict,
    outcome_effect: float,
    confidence_score: float,
    quality_grade: str = 'C'
) -> dict:
    """
    Generate text templates for multi-node combination interventions.

    Args:
        intervention_nodes: List of intervened variable names
        outcome_node: Name of the outcome variable
        intervention_pct_changes: Dict mapping node -> pct_change
        outcome_effect: Predicted percentage effect on outcome
        confidence_score: Confidence score 0-1
        quality_grade: Model quality grade

    Returns:
        Dictionary with platform-compatible fields
    """
    # Build path string for multiple nodes
    path_parts = [f"{node} -> {outcome_node}" for node in intervention_nodes]
    path_string = " | ".join(path_parts)

    # Determine outcome direction
    if outcome_effect < 0:
        outcome_direction = "bring down"
        outcome_action_summary = "reduce"
    else:
        outcome_direction = "bring up"
        outcome_action_summary = "increase"

    # Build intervention descriptions
    intervention_descriptions = []
    for node in intervention_nodes:
        pct = intervention_pct_changes.get(node, 0)
        action = "reduce" if pct < 0 else "increase"
        intervention_descriptions.append(f"{action} {node} by {abs(pct):.1f}%")

    interventions_text = " and ".join(intervention_descriptions)

    # Build summary
    summary = f"{interventions_text.capitalize()} to achieve {outcome_action_summary} in {outcome_node}"

    # Build recommendation detail for combinations
    levers_text = []
    for node in intervention_nodes:
        pct = intervention_pct_changes.get(node, 0)
        action = "decrease" if pct < 0 else "increase"
        levers_text.append(f"{action} {node} by {abs(pct):.2f}%")

    recommendation_detail = (
        f"To {outcome_direction} {outcome_node} by {round(outcome_effect, 2)}%, "
        f"the recommended combination is to {' and '.join(levers_text)}"
    )

    # Compute priority
    if confidence_score >= 0.7:
        priority = "High"
    elif confidence_score >= 0.5:
        priority = "Medium"
    else:
        priority = "Low"

    # Compute validity
    validity_map = {'A': 30, 'B': 28, 'C': 21, 'D': 14, 'F': 7}
    validity = validity_map.get(quality_grade, 21)

    # Confidence text
    if confidence_score >= 0.7:
        confidence_text = "High"
    elif confidence_score >= 0.5:
        confidence_text = "Medium"
    else:
        confidence_text = "Low"

    return {
        'details': [{'path': path_string}],
        'summary': summary,
        'priority': priority,
        'validity': validity,
        'confidence': confidence_text,
        'opportunityValue': None,
        'recommendationDetail': recommendation_detail
    }


class InterventionSearch:
    """
    Production-ready Intervention Search System.

    Finds optimal causal interventions with proper uncertainty quantification,
    model quality gating, and path sensitivity analysis.

    Example:
        >>> # Initialize with trained causal model
        >>> searcher = InterventionSearch(graph, ht_model)
        >>>
        >>> # Find interventions
        >>> results = searcher.find_interventions(
        ...     target_outcome='revenue',
        ...     target_change=+15,  # Increase by 15%
        ...     tolerance=3.0,      # Accept ±3%
        ...     max_candidates=10
        ... )
        >>>
        >>> # Get best recommendation (marked with best=True)
        >>> best = next(c for c in results['all_candidates'] if c['best'])
        >>> print(f"Intervene on: {best['nodes']}")
        >>> print(f"Expected effect: {best['actual_effect']:+.1f}%")
        >>> print(f"Summary: {best['summary']}")
        >>> print(f"Priority: {best['priority']}, Confidence: {best['confidence']}")
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        ht_model,  # Trained HT model from ht_categ.py
        n_simulations: int = 5000,
        strict_quality_mode: bool = False,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize Intervention Search system.

        Args:
            graph: Causal DAG (networkx DiGraph)
            ht_model: Trained HT model with regressors, baseline stats, etc.
            n_simulations: Number of Monte Carlo simulations (default: 5000, increased for narrower CIs)
            strict_quality_mode: If True, apply strict quality gates
            random_seed: Random seed for reproducibility
        """
        self.graph = graph
        self.ht_model = ht_model

        # Extract components from HT model
        self.regressors_dict = ht_model.regressors_dict
        self.baseline_stats = ht_model.baseline_stats
        self.model_metrics = getattr(ht_model, 'model_metrics', {})
        self.node_types = getattr(ht_model, 'node_types', {})
        self.edge_elasticities = getattr(ht_model, 'edge_elasticities', {})

        # Initialize components
        self.propagator = MonteCarloPropagator(
            graph=graph,
            regressors_dict=self.regressors_dict,
            baseline_stats=self.baseline_stats,
            model_metrics=self.model_metrics,
            node_types=self.node_types,
            n_simulations=n_simulations,
            random_seed=random_seed
        )

        self.quality_gate = QualityGate(
            min_r2_threshold=0.5,
            strict_mode=strict_quality_mode
        )

        self.path_analyzer = PathSensitivityAnalyzer(
            graph=graph,
            model_metrics=self.model_metrics,
            edge_elasticities=self.edge_elasticities
        )

        self.validator = InterventionValidator(
            baseline_stats=self.baseline_stats,
            strict_mode=strict_quality_mode
        )

        self.ranker = InterventionRanker(prioritize_safety=True)

        # Cache for quality summary
        self._quality_summary = None

    def find_interventions(
        self,
        target_outcome: str,
        target_change: float,
        candidate_nodes: Optional[List[str]] = None,
        tolerance: float = 3.0,
        max_intervention_pct: float = 30.0,
        allow_combinations: bool = False,
        max_candidates: int = 10,
        confidence_level: float = 0.90,
        min_model_quality: float = 0.5,
        verbose: bool = True
    ) -> Dict:
        """
        Find optimal interventions to achieve target outcome change.

        Args:
            target_outcome: Name of outcome variable to change
            target_change: Target percentage change (e.g., +10 for 10% increase)
            candidate_nodes: List of nodes to consider (None = all ancestors)
            tolerance: Acceptable error in percentage points (default: ±3%)
            max_intervention_pct: Maximum allowed intervention (default: ±30%)
            allow_combinations: Whether to test 2-node combinations
            max_candidates: Maximum number of candidates to return
            confidence_level: Confidence level for intervals (default: 0.90)
            min_model_quality: Minimum model R² to consider (default: 0.5)
            verbose: Print progress information

        Returns:
            Dictionary with:
                - all_candidates: All viable options (ranked), each containing:
                    - best: Boolean flag (True for top recommendation)
                    - details: List with path info [{'path': 'X -> Y'}]
                    - summary: Human-readable intervention summary
                    - priority: "High", "Medium", or "Low"
                    - validity: Days the recommendation is valid (int)
                    - confidence: "High", "Medium", or "Low" (text)
                    - confidence_score: Original numeric confidence (0-1)
                    - opportunityValue: Optional value (null by default)
                    - recommendationDetail: Detailed recommendation text
                - quality_report: Model quality assessment
                - path_analysis: Causal path analysis
                - summary: Summary statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 INTERVENTION SEARCH v2.0 (Production Grade)")
            print(f"{'='*70}")
            print(f"Target: {target_change:+.1f}% change in {target_outcome}")
            print(f"Tolerance: ±{tolerance}% points")
            print(f"Max intervention: ±{max_intervention_pct}%")
            print(f"Monte Carlo simulations: {self.propagator.n_simulations}")

        # Validate inputs
        if target_outcome not in self.graph.nodes():
            raise ValueError(f"Outcome node '{target_outcome}' not found in graph!")

        # Get candidate nodes
        if candidate_nodes is None:
            candidate_nodes = list(nx.ancestors(self.graph, target_outcome))
            if not candidate_nodes:
                raise ValueError(f"No ancestor nodes found for '{target_outcome}'!")

        if verbose:
            print(f"\n📊 Pre-flight checks...")
            print(f"   Candidate nodes: {len(candidate_nodes)}")

        # Quality assessment
        quality_summary = self.get_quality_summary()
        if verbose:
            print(f"   Overall model quality: {quality_summary['overall_grade']}")
            if quality_summary['low_quality_count'] > 0:
                print(f"   ⚠️  {quality_summary['low_quality_count']} low-quality models detected")

        # Search for single-node interventions
        if verbose:
            print(f"\n🔍 Searching {len(candidate_nodes)} candidates...")

        single_node_results = self._search_single_node_interventions(
            candidate_nodes=candidate_nodes,
            outcome_node=target_outcome,
            target_change=target_change,
            tolerance=tolerance,
            max_intervention_pct=max_intervention_pct,
            min_model_quality=min_model_quality,
            verbose=verbose
        )

        all_results = single_node_results

        # Search for combinations if requested
        if allow_combinations and len(candidate_nodes) >= 2:
            if verbose:
                print(f"\n🔗 Testing 2-node combinations...")

            combo_results = self._search_combination_interventions(
                candidate_nodes=candidate_nodes,
                outcome_node=target_outcome,
                target_change=target_change,
                tolerance=tolerance,
                max_intervention_pct=max_intervention_pct,
                max_combos=10,
                verbose=verbose
            )

            all_results.extend(combo_results)

        if not all_results:
            if verbose:
                print(f"\n❌ No feasible interventions found")
            return {
                'error': 'No feasible interventions found',
                'recommendation': 'Consider: (1) Relaxing constraints, (2) Improving model quality, (3) Different target',
                'quality_report': quality_summary
            }

        # Validate interventions
        if verbose:
            print(f"\n✅ Validating {len(all_results)} candidates...")

        validated_results = self.validator.filter_valid_interventions(
            all_results,
            require_feasible=True,
            require_safe=False  # Allow OOD with warning
        )

        if not validated_results:
            if verbose:
                print(f"   ⚠️  No interventions passed validation")
            validated_results = all_results  # Fall back to unvalidated

        # Rank interventions
        ranked_results = self.ranker.rank_interventions(
            validated_results,
            target_effect=target_change,
            tolerance=tolerance
        )

        # Get top candidates
        top_candidates = ranked_results[:max_candidates]

        # Compute summary statistics BEFORE enrichment (while confidence is still numeric)
        summary_stats = {
            'total_tested': len(all_results),
            'passed_validation': len(validated_results),
            'within_tolerance': sum(1 for r in ranked_results if r.get('within_tolerance', False)),
            'high_confidence': sum(1 for r in ranked_results if r.get('confidence', 0) >= 0.7),
            'target_achieved': top_candidates[0].get('within_tolerance', False) if top_candidates else False
        }

        # Enrich each candidate with platform-compatible fields
        for idx, candidate in enumerate(top_candidates):
            # Add 'best' flag - first candidate is best
            candidate['best'] = (idx == 0)

            # Get quality grade for validity calculation
            quality_info = candidate.get('quality', {})
            quality_grade = quality_info.get('quality_grade', 'C')
            confidence_score = candidate.get('confidence', 0.5)

            # Generate text fields based on intervention type
            if candidate.get('intervention_type') == 'single':
                intervention_node = candidate['nodes'][0]
                intervention_pct = candidate['required_pct_changes'].get(intervention_node, 0)
                outcome_effect = candidate.get('actual_effect', 0)

                text_fields = _generate_intervention_text_fields(
                    intervention_node=intervention_node,
                    outcome_node=target_outcome,
                    intervention_pct_change=intervention_pct,
                    outcome_effect=outcome_effect,
                    confidence_score=confidence_score,
                    quality_grade=quality_grade
                )
            else:
                # Combination intervention
                text_fields = _generate_combination_text_fields(
                    intervention_nodes=candidate['nodes'],
                    outcome_node=target_outcome,
                    intervention_pct_changes=candidate.get('required_pct_changes', {}),
                    outcome_effect=candidate.get('actual_effect', 0),
                    confidence_score=confidence_score,
                    quality_grade=quality_grade
                )

            # Add all text fields to candidate
            # Preserve numeric confidence as confidence_score, use 'confidence' for text
            candidate['confidence_score'] = candidate.get('confidence', 0.5)
            candidate['details'] = text_fields['details']
            candidate['summary'] = text_fields['summary']
            candidate['priority'] = text_fields['priority']
            candidate['validity'] = text_fields['validity']
            candidate['confidence'] = text_fields['confidence']  # Text version: "High", "Medium", "Low"
            candidate['opportunityValue'] = text_fields['opportunityValue']
            candidate['recommendationDetail'] = text_fields['recommendationDetail']

        # Analyze best intervention paths
        best = top_candidates[0] if top_candidates else None

        path_analysis = None
        if best is not None:
            intervention_node = best['nodes'][0] if best['nodes'] else None
            if intervention_node:
                path_analysis = self.path_analyzer.decompose_total_effect(
                    intervention_node,
                    target_outcome,
                    best.get('actual_effect', 0)
                )

        # Generate summary
        if verbose:
            self._print_results(best, top_candidates, target_change)

        # Return structure without duplication - 'best' flag indicates best intervention
        return {
            'all_candidates': top_candidates,
            'quality_report': quality_summary,
            'path_analysis': path_analysis,
            'summary': summary_stats
        }

    def _search_single_node_interventions(
        self,
        candidate_nodes: List[str],
        outcome_node: str,
        target_change: float,
        tolerance: float,
        max_intervention_pct: float,
        min_model_quality: float,
        verbose: bool
    ) -> List[Dict]:
        """Search for optimal single-node interventions"""
        results = []

        for node in candidate_nodes:
            if node == outcome_node:
                continue

            # Check for causal path
            if not nx.has_path(self.graph, node, outcome_node):
                continue

            # Check model quality
            if node in self.model_metrics:
                r2 = self.model_metrics[node].get('r2_score', 0)
                if r2 < min_model_quality:
                    continue

            if verbose:
                print(f"   Testing: {node}...", end=" ")

            # Define objective function for this node
            def objective_fn(pct_change):
                return self._simulate_single_intervention(node, pct_change, outcome_node)

            # Use adaptive search
            optimizer = AdaptiveGridSearch(
                objective_function=objective_fn,
                bounds=(-max_intervention_pct, max_intervention_pct),
                target_value=target_change,
                tolerance=tolerance,
                max_iterations=15
            )

            search_result = optimizer.search()

            if search_result.converged or abs(search_result.error_from_target) < tolerance * 2:
                # Get detailed simulation with uncertainty
                detailed_result = self._simulate_intervention_with_uncertainty(
                    {node: search_result.intervention_pct},
                    outcome_node
                )

                # Evaluate quality for this intervention path
                path_nodes = self._get_causal_path_nodes(node, outcome_node)
                quality_info = self.quality_gate.evaluate_path_quality(
                    path_nodes,
                    self.model_metrics,
                    self.baseline_stats
                )

                result_entry = {
                    'intervention_type': 'single',
                    'nodes': [node],
                    'required_pct_changes': {node: round(search_result.intervention_pct, 2)},
                    'actual_effect': round(search_result.predicted_effect, 2),
                    'error_from_target': round(search_result.error_from_target, 2),
                    'within_tolerance': search_result.error_from_target <= tolerance,
                    'ci_90': detailed_result['ci_90'],
                    'ci_50': detailed_result['ci_50'],
                    'prediction_uncertainty_std': detailed_result['std'],
                    'confidence': self._compute_confidence(
                        search_result.predicted_effect,
                        target_change,
                        detailed_result['std'],
                        [node]
                    ),
                    'quality': quality_info,  # FIX: Add quality metrics
                    'search_iterations': search_result.iterations
                }

                results.append(result_entry)

                if verbose:
                    status = "✓" if result_entry['within_tolerance'] else "~"
                    print(f"{status} {search_result.intervention_pct:+.1f}% → {search_result.predicted_effect:+.1f}%")
            else:
                if verbose:
                    print("✗ No solution found")

        return results

    def _search_combination_interventions(
        self,
        candidate_nodes: List[str],
        outcome_node: str,
        target_change: float,
        tolerance: float,
        max_intervention_pct: float,
        max_combos: int,
        verbose: bool
    ) -> List[Dict]:
        """Search for 2-node combination interventions"""
        from itertools import combinations

        results = []
        good_nodes = candidate_nodes[:10]  # Limit for performance

        for node1, node2 in list(combinations(good_nodes, 2))[:max_combos]:
            # Define objective for combination
            def objective_fn(intervention_dict):
                return self._simulate_single_intervention_dict(intervention_dict, outcome_node)

            optimizer = MultiNodeOptimizer(
                objective_function=objective_fn,
                node_bounds={
                    node1: (-max_intervention_pct, max_intervention_pct),
                    node2: (-max_intervention_pct, max_intervention_pct)
                },
                target_value=target_change,
                tolerance=tolerance,
                max_evaluations=50
            )

            opt_result = optimizer.search()

            if opt_result['converged'] or opt_result['error'] < tolerance * 2:
                detailed_result = self._simulate_intervention_with_uncertainty(
                    opt_result['intervention'],
                    outcome_node
                )

                # Evaluate quality for combination intervention
                # Use the shortest/strongest path among the two nodes
                path_nodes_1 = self._get_causal_path_nodes(node1, outcome_node)
                path_nodes_2 = self._get_causal_path_nodes(node2, outcome_node)
                # Choose path with better quality
                quality_1 = self.quality_gate.evaluate_path_quality(
                    path_nodes_1, self.model_metrics, self.baseline_stats
                )
                quality_2 = self.quality_gate.evaluate_path_quality(
                    path_nodes_2, self.model_metrics, self.baseline_stats
                )
                quality_info = quality_1 if quality_1['quality_score_geom_mean'] >= quality_2['quality_score_geom_mean'] else quality_2

                result_entry = {
                    'intervention_type': 'combination',
                    'nodes': [node1, node2],
                    'required_pct_changes': {
                        k: round(v, 2) for k, v in opt_result['intervention'].items()
                    },
                    'actual_effect': round(opt_result['predicted_effect'], 2),
                    'error_from_target': round(opt_result['error'], 2),
                    'within_tolerance': opt_result['error'] <= tolerance,
                    'ci_90': detailed_result['ci_90'],
                    'ci_50': detailed_result['ci_50'],
                    'prediction_uncertainty_std': detailed_result['std'],
                    'confidence': self._compute_confidence(
                        opt_result['predicted_effect'],
                        target_change,
                        detailed_result['std'],
                        [node1, node2]
                    ),
                    'quality': quality_info,  # FIX: Add quality metrics
                    'search_iterations': opt_result['iterations']
                }

                results.append(result_entry)

                if verbose:
                    print(f"   ✓ {node1} + {node2} → {opt_result['predicted_effect']:+.1f}%")

        return results

    def _simulate_single_intervention(
        self,
        node: str,
        pct_change: float,
        outcome_node: str
    ) -> float:
        """Simulate single intervention (returns mean effect)"""
        baseline_values = {n: self.baseline_stats.get(n, {}).get('mean', 0) for n in self.graph.nodes()}
        baseline_value = baseline_values[node]
        intervention_value = baseline_value * (1 + pct_change / 100)

        # Use propagator
        result = self.propagator.compare_interventions(
            initial_values=baseline_values,
            intervention_values={node: intervention_value},
            outcome_nodes=[outcome_node]
        )

        return result['outcome_effects'][outcome_node].get('pct_effect', 0)

    def _simulate_single_intervention_dict(
        self,
        intervention_dict: Dict[str, float],
        outcome_node: str
    ) -> float:
        """Simulate intervention from percentage dict"""
        baseline_values = {n: self.baseline_stats.get(n, {}).get('mean', 0) for n in self.graph.nodes()}

        intervention_values = {}
        for node, pct_change in intervention_dict.items():
            baseline_value = baseline_values[node]
            intervention_values[node] = baseline_value * (1 + pct_change / 100)

        result = self.propagator.compare_interventions(
            initial_values=baseline_values,
            intervention_values=intervention_values,
            outcome_nodes=[outcome_node]
        )

        return result['outcome_effects'][outcome_node].get('pct_effect', 0)

    def _simulate_intervention_with_uncertainty(
        self,
        intervention_pct_dict: Dict[str, float],
        outcome_node: str
    ) -> Dict:
        """Simulate with full uncertainty quantification"""
        baseline_values = {n: self.baseline_stats.get(n, {}).get('mean', 0) for n in self.graph.nodes()}

        intervention_values = {}
        for node, pct_change in intervention_pct_dict.items():
            baseline_value = baseline_values[node]
            intervention_values[node] = baseline_value * (1 + pct_change / 100)

        result = self.propagator.compare_interventions(
            initial_values=baseline_values,
            intervention_values=intervention_values,
            outcome_nodes=[outcome_node]
        )

        outcome_effect = result['outcome_effects'][outcome_node]

        return {
            'mean': outcome_effect.get('pct_effect', 0),
            'std': outcome_effect.get('absolute_effect_std', 0),
            'ci_90': outcome_effect.get('ci_90', (0, 0)),
            'ci_50': outcome_effect.get('ci_50', (0, 0))
        }

    def _compute_confidence(
        self,
        predicted_effect: float,
        target_effect: float,
        uncertainty_std: float,
        nodes: List[str]
    ) -> float:
        """Compute confidence score for intervention"""
        # Get model quality scores for nodes
        quality_scores = []
        for node in nodes:
            if node in self.model_metrics:
                metrics = self.model_metrics[node]
                if metrics.get('model_type') == 'regression':
                    r2 = metrics.get('r2_score', 0.5)
                    quality_scores.append(r2)

        return self.quality_gate.compute_intervention_reliability_score(
            nodes,
            'outcome',  # Placeholder
            self.model_metrics,
            self.baseline_stats,
            predicted_effect,
            target_effect,
            uncertainty_std
        )

    def get_quality_summary(self) -> Dict:
        """Get model quality summary"""
        if self._quality_summary is None:
            self._quality_summary = generate_quality_summary(
                self.model_metrics,
                self.baseline_stats
            )
        return self._quality_summary

    def _get_causal_path_nodes(self, source: str, target: str) -> List[str]:
        """
        Extract causal path nodes from source to target.

        For simplicity, returns shortest path. In production,
        might want to consider all paths or strongest path.

        Args:
            source: Intervention node
            target: Outcome node

        Returns:
            List of nodes in the causal path
        """
        try:
            # Get shortest path
            path = nx.shortest_path(self.graph, source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists, return just the nodes
            return [source, target]

    def _print_results(self, best, all_candidates, target_change):
        """Print results summary"""
        print(f"\n{'='*70}")
        print(f"✅ SEARCH COMPLETE")
        print(f"{'='*70}")

        if best is None:
            print("\n❌ No interventions found")
            return

        print(f"\nBest Intervention:")
        print(f"   Type: {best['intervention_type']}")
        print(f"   Variables: {', '.join(best['nodes'])}")
        for node, pct in best['required_pct_changes'].items():
            print(f"   └─ {node}: {pct:+.2f}%")
        print(f"\n   Predicted Effect: {best['actual_effect']:+.1f}% (target: {target_change:+.1f}%)")
        print(f"   90% Confidence Interval: [{best['ci_90'][0]:+.1f}%, {best['ci_90'][1]:+.1f}%]")
        print(f"   50% Confidence Interval: [{best['ci_50'][0]:+.1f}%, {best['ci_50'][1]:+.1f}%]")
        print(f"   Confidence Score: {best['confidence_score']:.0%}")

        status = "✅ APPROVED" if best.get('within_tolerance', False) else "⚠️  CAUTION"
        print(f"\n   Status: {status}")

        print(f"\n   Total Candidates Found: {len(all_candidates)}")
        print(f"{'='*70}\n")
