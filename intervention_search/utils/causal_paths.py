"""
Causal Path Analysis Utilities

Tools for enumerating, analyzing, and scoring causal paths through a DAG.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CausalPath:
    """Represents a causal path through the graph"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    length: int

    def __post_init__(self):
        self.length = len(self.edges)

    def __repr__(self):
        return " → ".join(self.nodes)

    def __hash__(self):
        return hash(tuple(self.nodes))

    def __eq__(self, other):
        return self.nodes == other.nodes


def enumerate_all_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_length: Optional[int] = None
) -> List[CausalPath]:
    """
    Enumerate all simple paths from source to target in a DAG.

    Args:
        graph: Directed acyclic graph
        source: Source node
        target: Target node
        max_length: Maximum path length (None = no limit)

    Returns:
        List of CausalPath objects
    """
    if not nx.has_path(graph, source, target):
        return []

    paths = []
    cutoff = max_length if max_length is not None else None

    for node_path in nx.all_simple_paths(graph, source, target, cutoff=cutoff):
        edges = [(node_path[i], node_path[i+1]) for i in range(len(node_path)-1)]
        paths.append(CausalPath(nodes=node_path, edges=edges, length=len(edges)))

    return paths


def compute_path_quality_score(
    path: CausalPath,
    model_metrics: Dict[str, Dict],
    graph: nx.DiGraph
) -> Dict:
    """
    Compute comprehensive quality metrics for a causal path.

    Quality factors:
    1. Model quality (R² scores along path)
    2. Path length (shorter is better)
    3. Effect attenuation (longer paths have weaker effects)

    Args:
        path: CausalPath object
        model_metrics: Dictionary of model metrics for each node
        graph: Causal graph

    Returns:
        Dictionary with quality scores
    """
    # Collect R² scores for nodes with models (excluding first node)
    r2_scores = []
    rmse_scores = []

    for node in path.nodes[1:]:  # Skip source node
        if node in model_metrics:
            metrics = model_metrics[node]
            if metrics.get('model_type') == 'regression':
                r2 = metrics.get('r2_score', 0.5)
                rmse = metrics.get('rmse', 1.0)
                r2_scores.append(r2)
                rmse_scores.append(rmse)

    # Quality score 1: Minimum R² (weakest link)
    min_r2 = min(r2_scores) if r2_scores else 0.5

    # Quality score 2: Mean R² (average quality)
    mean_r2 = np.mean(r2_scores) if r2_scores else 0.5

    # Quality score 3: Geometric mean R² (penalizes weak links heavily)
    geom_mean_r2 = np.prod(r2_scores) ** (1.0 / len(r2_scores)) if r2_scores else 0.5

    # Quality score 4: Path length penalty
    length_penalty = 0.9 ** path.length  # Exponential decay

    # Quality score 5: Accumulated RMSE (uncertainty compounds)
    accumulated_rmse = np.sqrt(sum(rmse**2 for rmse in rmse_scores)) if rmse_scores else 1.0

    # Combined quality score (geometric mean to penalize any weak component)
    component_scores = [min_r2, mean_r2, geom_mean_r2, length_penalty]
    overall_quality = np.prod(component_scores) ** (1.0 / len(component_scores))

    return {
        'min_r2': min_r2,
        'mean_r2': mean_r2,
        'geom_mean_r2': geom_mean_r2,
        'length': path.length,
        'length_penalty': length_penalty,
        'accumulated_rmse': accumulated_rmse,
        'overall_quality': overall_quality,
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores
    }


def rank_paths_by_quality(
    paths: List[CausalPath],
    model_metrics: Dict[str, Dict],
    graph: nx.DiGraph,
    min_quality_threshold: float = 0.3
) -> List[Tuple[CausalPath, Dict]]:
    """
    Rank causal paths by quality score.

    Args:
        paths: List of CausalPath objects
        model_metrics: Model metrics dictionary
        graph: Causal graph
        min_quality_threshold: Minimum quality to include

    Returns:
        List of (path, quality_dict) tuples, sorted by quality
    """
    scored_paths = []

    for path in paths:
        quality = compute_path_quality_score(path, model_metrics, graph)
        if quality['overall_quality'] >= min_quality_threshold:
            scored_paths.append((path, quality))

    # Sort by overall quality (descending)
    scored_paths.sort(key=lambda x: x[1]['overall_quality'], reverse=True)

    return scored_paths


def identify_critical_nodes(
    paths: List[CausalPath],
    model_metrics: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Identify nodes that are critical (appear in many paths or have low quality).

    Args:
        paths: List of causal paths
        model_metrics: Model metrics dictionary

    Returns:
        Dictionary mapping node to criticality info
    """
    node_info = {}

    for path in paths:
        for node in path.nodes:
            if node not in node_info:
                node_info[node] = {
                    'appearance_count': 0,
                    'paths_through': [],
                    'is_bottleneck': False,
                    'quality': 1.0
                }

            node_info[node]['appearance_count'] += 1
            node_info[node]['paths_through'].append(path)

            # Get model quality
            if node in model_metrics:
                metrics = model_metrics[node]
                if metrics.get('model_type') == 'regression':
                    r2 = metrics.get('r2_score', 1.0)
                    node_info[node]['quality'] = min(node_info[node]['quality'], r2)

    # Identify bottlenecks (nodes that appear in ALL paths)
    total_paths = len(paths)
    for node, info in node_info.items():
        if info['appearance_count'] == total_paths and total_paths > 1:
            info['is_bottleneck'] = True

    return node_info


def compute_path_elasticity(
    path: CausalPath,
    edge_elasticities: Dict[Tuple[str, str], float]
) -> float:
    """
    Compute the overall elasticity along a causal path.

    For a path X → Y → Z with elasticities ε_XY and ε_YZ:
    Overall elasticity ≈ ε_XY × ε_YZ

    Args:
        path: CausalPath object
        edge_elasticities: Dictionary mapping edges to elasticities

    Returns:
        Combined elasticity along the path
    """
    elasticity = 1.0

    for edge in path.edges:
        edge_elast = edge_elasticities.get(edge, 1.0)
        elasticity *= edge_elast

    return elasticity


def filter_redundant_paths(
    paths: List[CausalPath],
    similarity_threshold: float = 0.8
) -> List[CausalPath]:
    """
    Filter out redundant/highly similar paths using Jaccard similarity.

    Args:
        paths: List of CausalPath objects
        similarity_threshold: Jaccard similarity threshold for filtering

    Returns:
        Filtered list of non-redundant paths
    """
    if not paths:
        return []

    filtered = [paths[0]]  # Always keep the first path

    for path in paths[1:]:
        path_nodes = set(path.nodes)

        # Check similarity with already selected paths
        is_redundant = False
        for selected_path in filtered:
            selected_nodes = set(selected_path.nodes)

            # Jaccard similarity
            intersection = len(path_nodes & selected_nodes)
            union = len(path_nodes | selected_nodes)
            similarity = intersection / union if union > 0 else 0

            if similarity > similarity_threshold:
                is_redundant = True
                break

        if not is_redundant:
            filtered.append(path)

    return filtered


def find_most_reliable_path(
    source: str,
    target: str,
    graph: nx.DiGraph,
    model_metrics: Dict[str, Dict],
    max_paths: int = 10
) -> Optional[Tuple[CausalPath, Dict]]:
    """
    Find the single most reliable causal path from source to target.

    Args:
        source: Source node
        target: Target node
        graph: Causal graph
        model_metrics: Model metrics dictionary
        max_paths: Maximum number of paths to consider

    Returns:
        Tuple of (best_path, quality_metrics) or None
    """
    # Enumerate paths
    all_paths = enumerate_all_paths(graph, source, target, max_length=6)

    if not all_paths:
        return None

    # Rank by quality
    ranked_paths = rank_paths_by_quality(all_paths[:max_paths], model_metrics, graph)

    if not ranked_paths:
        return None

    return ranked_paths[0]  # Return best path


def decompose_effect_by_path(
    intervention_node: str,
    outcome_node: str,
    graph: nx.DiGraph,
    model_metrics: Dict[str, Dict]
) -> Dict:
    """
    Decompose the total causal effect into contributions from individual paths.

    Args:
        intervention_node: Node being intervened on
        outcome_node: Outcome node
        graph: Causal graph
        model_metrics: Model metrics dictionary

    Returns:
        Dictionary with path decomposition analysis
    """
    # Enumerate all paths
    paths = enumerate_all_paths(graph, intervention_node, outcome_node)

    # Score each path
    scored_paths = rank_paths_by_quality(paths, model_metrics, graph, min_quality_threshold=0.0)

    # Compute statistics
    total_paths = len(paths)
    high_quality_paths = len([p for p, q in scored_paths if q['overall_quality'] >= 0.7])

    # Identify critical nodes
    critical_nodes = identify_critical_nodes(paths, model_metrics)
    bottlenecks = [node for node, info in critical_nodes.items() if info['is_bottleneck']]

    return {
        'total_paths': total_paths,
        'high_quality_paths': high_quality_paths,
        'path_quality_ratio': high_quality_paths / total_paths if total_paths > 0 else 0,
        'scored_paths': scored_paths,
        'bottlenecks': bottlenecks,
        'critical_nodes': critical_nodes
    }
