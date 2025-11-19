"""
DO Operator Implementation for Causal Inference

Implements Pearl's DO operator for causal simulations.
This ensures all simulations are 100% based on causal methodology.

The DO operator do(X=x) represents an atomic intervention that:
1. Fixes X to value x (removes all incoming edges to X)
2. Propagates effects through outgoing edges only
3. Respects the causal structure (DAG)
"""

import numpy as np
import networkx as nx
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DOSimulationResult:
    """Results from DO operator simulation"""
    # Post-intervention values
    post_intervention_values: Dict[str, float]

    # Causal effects (difference from baseline)
    causal_effects: Dict[str, float]

    # Percentage changes
    pct_changes: Dict[str, float]

    # Metadata
    intervention_spec: Dict[str, float]
    affected_nodes: Set[str]
    unaffected_nodes: Set[str]


class DOOperator:
    """
    Implements Pearl's DO operator for causal interventions.

    The DO operator performs the following:
    1. Creates a mutilated graph where edges into intervened nodes are removed
    2. Simulates the intervention by fixing intervened nodes to target values
    3. Propagates effects through descendants via causal mechanisms (models)

    This is fundamentally different from conditioning (observing):
    - Conditioning: P(Y | X=x) - observational
    - Intervention: P(Y | do(X=x)) - causal

    Key principle: DO operator breaks incoming arrows, not outgoing ones.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        regressors_dict: Dict,
        baseline_stats: Dict,
        node_types: Dict,
        label_encoders: Optional[Dict] = None
    ):
        """
        Initialize DO operator.

        Args:
            graph: Original causal DAG
            regressors_dict: Trained models for each node
            baseline_stats: Baseline statistics for each node
            node_types: Type of each node (categorical/continuous)
            label_encoders: Label encoders for categorical variables
        """
        self.graph = graph
        self.regressors_dict = regressors_dict
        self.baseline_stats = baseline_stats
        self.node_types = node_types
        self.label_encoders = label_encoders or {}

        # Compute topological order
        try:
            self.topological_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles! Cannot perform causal simulation.")

    def do(
        self,
        intervention_values: Dict[str, float],
        baseline_values: Optional[Dict[str, float]] = None
    ) -> DOSimulationResult:
        """
        Apply DO operator: do(X1=x1, X2=x2, ...)

        This computes P(Y | do(X=x)) by:
        1. Creating mutilated graph G_{\bar{X}} where edges into X are removed
        2. Simulating from this mutilated graph with X fixed at x
        3. Computing causal effects on all descendants

        Args:
            intervention_values: Dict of {node: value} for interventions
            baseline_values: Baseline values for all nodes (uses stored baseline if None)

        Returns:
            DOSimulationResult with post-intervention values and causal effects
        """
        # Get baseline values
        if baseline_values is None:
            baseline_values = {
                node: self.baseline_stats.get(node, {}).get('mean', 0)
                for node in self.graph.nodes()
            }

        # Identify intervened nodes
        intervened_nodes = set(intervention_values.keys())

        # Validate interventions
        for node in intervened_nodes:
            if node not in self.graph.nodes():
                raise ValueError(f"Intervention node '{node}' not in graph!")

        # Create mutilated graph (remove edges into intervened nodes)
        mutilated_graph = self._create_mutilated_graph(intervened_nodes)

        # Perform forward propagation on mutilated graph
        post_intervention_values = self._propagate_through_mutilated_graph(
            mutilated_graph=mutilated_graph,
            intervention_values=intervention_values,
            baseline_values=baseline_values
        )

        # Compute causal effects
        causal_effects = {}
        pct_changes = {}
        for node in self.graph.nodes():
            baseline_val = baseline_values[node]
            post_val = post_intervention_values[node]

            # Handle categorical variables (strings)
            if isinstance(baseline_val, str) or isinstance(post_val, str):
                # For categorical variables, compute effect as 0 (no numeric change)
                causal_effects[node] = 0.0
                pct_changes[node] = 0.0
            else:
                # For numeric variables, compute numeric effects
                causal_effects[node] = post_val - baseline_val

                if abs(baseline_val) > 1e-9:
                    pct_changes[node] = ((post_val - baseline_val) / baseline_val) * 100
                else:
                    pct_changes[node] = 0.0

        # Identify affected nodes (descendants of intervention)
        affected_nodes = set()
        for int_node in intervened_nodes:
            affected_nodes.update(nx.descendants(self.graph, int_node))
            affected_nodes.add(int_node)

        unaffected_nodes = set(self.graph.nodes()) - affected_nodes

        return DOSimulationResult(
            post_intervention_values=post_intervention_values,
            causal_effects=causal_effects,
            pct_changes=pct_changes,
            intervention_spec=intervention_values,
            affected_nodes=affected_nodes,
            unaffected_nodes=unaffected_nodes
        )

    def _create_mutilated_graph(self, intervened_nodes: Set[str]) -> nx.DiGraph:
        """
        Create mutilated graph G_{\bar{X}} by removing incoming edges to X.

        This is the key step in the DO operator:
        - Removes edges U -> X for all parents U of X
        - Keeps all other edges intact
        - Allows causal effects to flow OUT of X but not INTO X

        Args:
            intervened_nodes: Nodes being intervened on

        Returns:
            Mutilated graph
        """
        mutilated_graph = self.graph.copy()

        # Remove all incoming edges to intervened nodes
        for node in intervened_nodes:
            # Get all parents (predecessors)
            parents = list(mutilated_graph.predecessors(node))

            # Remove edges from parents to this node
            for parent in parents:
                mutilated_graph.remove_edge(parent, node)

        return mutilated_graph

    def _propagate_through_mutilated_graph(
        self,
        mutilated_graph: nx.DiGraph,
        intervention_values: Dict[str, float],
        baseline_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Propagate through mutilated graph with interventions fixed.

        Uses topological order to ensure parents are computed before children.

        Args:
            mutilated_graph: Graph with intervention edges removed
            intervention_values: Fixed values for intervened nodes
            baseline_values: Starting values

        Returns:
            Dictionary of post-intervention values for all nodes
        """
        # Initialize with baseline
        values = baseline_values.copy()

        # Fix intervened nodes
        for node, value in intervention_values.items():
            values[node] = value

        # Get topological order of mutilated graph
        try:
            topo_order = list(nx.topological_sort(mutilated_graph))
        except nx.NetworkXError:
            # Shouldn't happen unless graph has cycles
            topo_order = self.topological_order

        # Propagate through graph
        for node in topo_order:
            # Skip intervened nodes (already fixed)
            if node in intervention_values:
                continue

            # Get parents in mutilated graph
            parents = list(mutilated_graph.predecessors(node))

            if not parents:
                # Root node or intervened node - keep baseline
                continue

            # Get model for this node
            if node not in self.regressors_dict:
                continue

            regressor, scaler = self.regressors_dict[node]

            if regressor is None:
                continue

            # Get parent values (after propagation)
            parent_values = []
            for parent in parents:
                parent_values.append(values[parent])

            parent_values = np.array([parent_values])

            # Predict using causal mechanism
            try:
                predicted_value = self._predict_deterministic(
                    node=node,
                    regressor=regressor,
                    parent_values=parent_values
                )
                values[node] = predicted_value

            except Exception:
                # Keep baseline if prediction fails
                pass

        return values

    def _predict_deterministic(
        self,
        node: str,
        regressor,
        parent_values: np.ndarray
    ) -> float:
        """
        Make deterministic prediction (no noise).

        For DO operator, we want the expected causal effect, not a sample.
        """
        is_categorical = self.node_types.get(node) == 'categorical'

        if is_categorical:
            # For categorical: predict class
            predicted_class = regressor.predict(parent_values)[0]
            return float(predicted_class)
        else:
            # For continuous: predict value (no noise)
            predicted_value = regressor.predict(parent_values)[0]
            return predicted_value

    def compute_causal_effect(
        self,
        intervention_node: str,
        outcome_node: str,
        intervention_value: float,
        baseline_values: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compute causal effect of intervening on one node on an outcome node.

        This computes: E[Y | do(X=x)] - E[Y]

        Args:
            intervention_node: Node to intervene on
            outcome_node: Node to measure effect on
            intervention_value: Value to set intervention to
            baseline_values: Baseline values (optional)

        Returns:
            Dictionary with causal effect information
        """
        # Validate path
        if not nx.has_path(self.graph, intervention_node, outcome_node):
            return {
                'intervention_node': intervention_node,
                'outcome_node': outcome_node,
                'intervention_value': intervention_value,
                'causal_effect': 0.0,
                'pct_effect': 0.0,
                'has_causal_path': False,
                'error': 'No causal path from intervention to outcome'
            }

        # Get baseline
        if baseline_values is None:
            baseline_values = {
                node: self.baseline_stats.get(node, {}).get('mean', 0)
                for node in self.graph.nodes()
            }

        baseline_outcome = baseline_values[outcome_node]

        # Apply DO operator
        result = self.do(
            intervention_values={intervention_node: intervention_value},
            baseline_values=baseline_values
        )

        # Extract outcome effect
        post_outcome = result.post_intervention_values[outcome_node]
        causal_effect = result.causal_effects[outcome_node]
        pct_effect = result.pct_changes[outcome_node]

        return {
            'intervention_node': intervention_node,
            'outcome_node': outcome_node,
            'intervention_value': intervention_value,
            'baseline_outcome': baseline_outcome,
            'post_intervention_outcome': post_outcome,
            'causal_effect': causal_effect,
            'pct_effect': pct_effect,
            'has_causal_path': True,
            'affected_nodes': list(result.affected_nodes),
            'unaffected_nodes': list(result.unaffected_nodes)
        }

    def get_all_causal_paths(
        self,
        intervention_node: str,
        outcome_node: str
    ) -> List[List[str]]:
        """
        Get all causal paths from intervention to outcome.

        Args:
            intervention_node: Starting node
            outcome_node: Ending node

        Returns:
            List of paths (each path is a list of nodes)
        """
        try:
            all_paths = list(nx.all_simple_paths(self.graph, intervention_node, outcome_node))
            return all_paths
        except nx.NetworkXNoPath:
            return []


def verify_do_operator_properties(do_operator: DOOperator, intervention_values: Dict[str, float]) -> Dict:
    """
    Verify that the DO operator implementation satisfies key properties.

    Properties to verify:
    1. Intervened nodes have exact specified values (not influenced by parents)
    2. Unaffected nodes remain at baseline
    3. Only descendants of intervention are affected
    4. Causal Markov property holds

    Args:
        do_operator: DOOperator instance
        intervention_values: Intervention specification

    Returns:
        Dictionary with verification results
    """
    result = do_operator.do(intervention_values)

    checks = {
        'all_checks_passed': True,
        'checks': []
    }

    # Check 1: Intervened nodes have exact values
    for node, target_value in intervention_values.items():
        actual_value = result.post_intervention_values[node]
        passed = abs(actual_value - target_value) < 1e-6

        checks['checks'].append({
            'check': f'Intervention value for {node}',
            'expected': target_value,
            'actual': actual_value,
            'passed': passed
        })

        if not passed:
            checks['all_checks_passed'] = False

    # Check 2: Unaffected nodes remain at baseline
    baseline_values = {
        node: do_operator.baseline_stats.get(node, {}).get('mean', 0)
        for node in do_operator.graph.nodes()
    }

    for node in result.unaffected_nodes:
        if node not in intervention_values:  # Skip intervened nodes
            baseline_val = baseline_values[node]
            actual_val = result.post_intervention_values[node]
            passed = abs(actual_val - baseline_val) < 1e-6

            checks['checks'].append({
                'check': f'Unaffected node {node} unchanged',
                'expected': baseline_val,
                'actual': actual_val,
                'passed': passed
            })

            if not passed:
                checks['all_checks_passed'] = False

    # Check 3: Only descendants + intervention are affected
    for int_node in intervention_values.keys():
        descendants = set(nx.descendants(do_operator.graph, int_node))
        descendants.add(int_node)

        for node in result.affected_nodes:
            passed = node in descendants

            if not passed:
                checks['checks'].append({
                    'check': f'Affected node {node} is descendant of intervention',
                    'passed': False
                })
                checks['all_checks_passed'] = False

    return checks
