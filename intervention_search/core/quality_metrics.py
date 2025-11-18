"""
Model Quality Metrics and Gating

Tools for evaluating model quality and gating intervention recommendations based on
the reliability of the underlying causal models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelQualityReport:
    """Container for model quality assessment"""
    node: str
    model_type: str  # 'regression' or 'classification'
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    quality_grade: str = 'Unknown'  # A, B, C, D, F
    quality_score: float = 0.0  # 0-1 normalized score
    is_reliable: bool = False
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class QualityGate:
    """
    Quality gating system for intervention reliability assessment.

    Implements the principle: "Only recommend interventions where we have
    high confidence in the underlying causal models."
    """

    def __init__(
        self,
        min_r2_threshold: float = 0.5,
        min_rmse_relative: float = 0.2,  # RMSE < 20% of std
        strict_mode: bool = False
    ):
        """
        Initialize quality gate.

        Args:
            min_r2_threshold: Minimum R² for regression models
            min_rmse_relative: Maximum RMSE relative to std
            strict_mode: If True, apply stricter thresholds
        """
        self.min_r2_threshold = min_r2_threshold
        self.min_rmse_relative = min_rmse_relative
        self.strict_mode = strict_mode

        if strict_mode:
            self.min_r2_threshold = 0.7
            self.min_rmse_relative = 0.15

    def evaluate_model_quality(
        self,
        node: str,
        metrics: Dict,
        baseline_std: Optional[float] = None
    ) -> ModelQualityReport:
        """
        Evaluate quality of a single model.

        Args:
            node: Node name
            metrics: Model metrics dictionary
            baseline_std: Baseline standard deviation for relative RMSE

        Returns:
            ModelQualityReport with assessment
        """
        model_type = metrics.get('model_type', 'unknown')
        report = ModelQualityReport(node=node, model_type=model_type)

        if model_type == 'regression':
            r2 = metrics.get('r2_score', 0.0)
            rmse = metrics.get('rmse', float('inf'))

            report.r2_score = r2
            report.rmse = rmse

            # Compute quality score (0-1)
            quality_score = r2

            # Penalize high RMSE relative to baseline std
            if baseline_std is not None and baseline_std > 0:
                relative_rmse = rmse / baseline_std
                if relative_rmse > self.min_rmse_relative:
                    rmse_penalty = min(1.0, (relative_rmse - self.min_rmse_relative) / 0.5)
                    quality_score *= (1 - rmse_penalty * 0.5)

            report.quality_score = quality_score

            # Assign grade
            if quality_score >= 0.9:
                report.quality_grade = 'A'
                report.is_reliable = True
            elif quality_score >= 0.7:
                report.quality_grade = 'B'
                report.is_reliable = True
            elif quality_score >= 0.5:
                report.quality_grade = 'C'
                report.is_reliable = not self.strict_mode
            elif quality_score >= 0.3:
                report.quality_grade = 'D'
                report.is_reliable = False
                report.warnings.append(f"Low R² ({r2:.2f}), predictions may be unreliable")
            else:
                report.quality_grade = 'F'
                report.is_reliable = False
                report.warnings.append(f"Very low R² ({r2:.2f}), model is not trustworthy")

            # Additional warnings
            if baseline_std is not None and baseline_std > 0:
                relative_rmse = rmse / baseline_std
                if relative_rmse > 0.5:
                    report.warnings.append(
                        f"High prediction error: RMSE = {relative_rmse:.1%} of baseline std"
                    )

        elif model_type == 'classification':
            accuracy = metrics.get('accuracy', 0.0)
            report.accuracy = accuracy
            report.quality_score = accuracy

            # Assign grade
            if accuracy >= 0.9:
                report.quality_grade = 'A'
                report.is_reliable = True
            elif accuracy >= 0.75:
                report.quality_grade = 'B'
                report.is_reliable = True
            elif accuracy >= 0.6:
                report.quality_grade = 'C'
                report.is_reliable = not self.strict_mode
            else:
                report.quality_grade = 'D'
                report.is_reliable = False
                report.warnings.append(f"Low accuracy ({accuracy:.2f})")

        return report

    def evaluate_path_quality(
        self,
        path_nodes: List[str],
        model_metrics: Dict[str, Dict],
        baseline_stats: Dict[str, Dict]
    ) -> Dict:
        """
        Evaluate quality of all models along a causal path.

        Args:
            path_nodes: List of nodes in the path
            model_metrics: Model metrics for all nodes
            baseline_stats: Baseline statistics for all nodes

        Returns:
            Dictionary with path quality assessment
        """
        path_reports = []

        for node in path_nodes[1:]:  # Skip first node (intervention)
            if node in model_metrics:
                baseline_std = baseline_stats.get(node, {}).get('std')
                report = self.evaluate_model_quality(
                    node, model_metrics[node], baseline_std
                )
                path_reports.append(report)

        if not path_reports:
            return {
                'path': ' → '.join(path_nodes),
                'quality_grade': 'Unknown',
                'quality_score_min': 0.5,
                'quality_score_mean': 0.5,
                'quality_score_geom_mean': 0.5,
                'is_reliable': False,
                'weakest_link': None,
                'model_reports': [],
                'warnings': ['No models found in path']
            }

        # Weakest link principle
        quality_scores = [r.quality_score for r in path_reports]
        min_quality = min(quality_scores)
        mean_quality = np.mean(quality_scores)
        geom_mean_quality = np.prod(quality_scores) ** (1.0 / len(quality_scores))

        # Find weakest link
        weakest_idx = np.argmin(quality_scores)
        weakest_link = path_reports[weakest_idx]

        # Overall grade (based on geometric mean to penalize weak links)
        if geom_mean_quality >= 0.8:
            overall_grade = 'A'
        elif geom_mean_quality >= 0.6:
            overall_grade = 'B'
        elif geom_mean_quality >= 0.4:
            overall_grade = 'C'
        elif geom_mean_quality >= 0.2:
            overall_grade = 'D'
        else:
            overall_grade = 'F'

        # Reliability check
        all_reliable = all(r.is_reliable for r in path_reports)

        # Collect warnings
        all_warnings = []
        for report in path_reports:
            for warning in report.warnings:
                all_warnings.append(f"{report.node}: {warning}")

        return {
            'path': ' → '.join(path_nodes),
            'quality_grade': overall_grade,
            'quality_score_min': min_quality,
            'quality_score_mean': mean_quality,
            'quality_score_geom_mean': geom_mean_quality,
            'is_reliable': all_reliable,
            'weakest_link': {
                'node': weakest_link.node,
                'grade': weakest_link.quality_grade,
                'score': weakest_link.quality_score
            },
            'model_reports': path_reports,
            'warnings': all_warnings
        }

    def filter_interventions_by_quality(
        self,
        interventions: List[Dict],
        model_metrics: Dict[str, Dict],
        baseline_stats: Dict[str, Dict],
        min_quality_score: float = 0.5
    ) -> List[Dict]:
        """
        Filter intervention recommendations based on model quality.

        Args:
            interventions: List of intervention dictionaries
            model_metrics: Model metrics
            baseline_stats: Baseline statistics
            min_quality_score: Minimum quality score to pass

        Returns:
            Filtered list of interventions with quality scores added
        """
        filtered = []

        for intervention in interventions:
            # Get nodes involved (depends on structure)
            nodes = intervention.get('nodes', [])

            if not nodes:
                continue

            # Evaluate quality
            path_quality = self.evaluate_path_quality(nodes, model_metrics, baseline_stats)

            # Add quality info to intervention
            intervention['quality'] = path_quality

            # Filter by threshold
            if path_quality['quality_score_geom_mean'] >= min_quality_score:
                filtered.append(intervention)

        return filtered

    def compute_intervention_reliability_score(
        self,
        intervention_nodes: List[str],
        outcome_node: str,
        model_metrics: Dict[str, Dict],
        baseline_stats: Dict[str, Dict],
        predicted_effect: float,
        target_effect: float,
        uncertainty_std: float
    ) -> float:
        """
        Compute overall reliability score for an intervention recommendation.

        Combines:
        1. Model quality along the causal path
        2. Accuracy (closeness to target)
        3. Precision (low uncertainty)

        Args:
            intervention_nodes: Nodes being intervened on
            outcome_node: Target outcome
            model_metrics: Model metrics
            baseline_stats: Baseline stats
            predicted_effect: Predicted effect size
            target_effect: Target effect size
            uncertainty_std: Prediction uncertainty

        Returns:
            Reliability score (0-1)
        """
        # Component 1: Model quality (use path from first intervention node to outcome)
        path_nodes = intervention_nodes + [outcome_node]
        path_quality = self.evaluate_path_quality(path_nodes, model_metrics, baseline_stats)
        quality_component = path_quality['quality_score_geom_mean']

        # Component 2: Accuracy
        if abs(target_effect) > 1e-9:
            error_pct = abs(predicted_effect - target_effect) / abs(target_effect)
            accuracy_component = np.exp(-error_pct)
        else:
            accuracy_component = 0.5

        # Component 3: Precision
        if abs(predicted_effect) > 1e-9:
            cv = uncertainty_std / abs(predicted_effect)
            precision_component = np.exp(-cv)
        else:
            precision_component = 0.5

        # Weighted combination (model quality is most important)
        reliability = (
            0.5 * quality_component +
            0.25 * accuracy_component +
            0.25 * precision_component
        )

        return np.clip(reliability, 0.0, 1.0)


def generate_quality_summary(
    model_metrics: Dict[str, Dict],
    baseline_stats: Dict[str, Dict]
) -> Dict:
    """
    Generate a summary report of model quality across all nodes.

    Args:
        model_metrics: Model metrics dictionary
        baseline_stats: Baseline statistics

    Returns:
        Summary dictionary
    """
    gate = QualityGate()

    regression_scores = []
    classification_scores = []
    reports = []

    for node, metrics in model_metrics.items():
        baseline_std = baseline_stats.get(node, {}).get('std')
        report = gate.evaluate_model_quality(node, metrics, baseline_std)
        reports.append(report)

        if report.model_type == 'regression' and report.r2_score is not None:
            regression_scores.append(report.r2_score)
        elif report.model_type == 'classification' and report.accuracy is not None:
            classification_scores.append(report.accuracy)

    # Grade distribution
    grade_counts = {}
    for report in reports:
        grade = report.quality_grade
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    # Identify problematic models
    low_quality_models = [r for r in reports if r.quality_grade in ['D', 'F']]
    high_quality_models = [r for r in reports if r.quality_grade in ['A', 'B']]

    return {
        'total_models': len(reports),
        'regression_models': len(regression_scores),
        'classification_models': len(classification_scores),
        'mean_r2': np.mean(regression_scores) if regression_scores else None,
        'median_r2': np.median(regression_scores) if regression_scores else None,
        'grade_distribution': grade_counts,
        'high_quality_count': len(high_quality_models),
        'low_quality_count': len(low_quality_models),
        'low_quality_nodes': [r.node for r in low_quality_models],
        'overall_grade': _compute_overall_grade(reports)
    }


def _compute_overall_grade(reports: List[ModelQualityReport]) -> str:
    """Compute overall grade for the model suite"""
    if not reports:
        return 'Unknown'

    scores = [r.quality_score for r in reports]
    geom_mean = np.prod(scores) ** (1.0 / len(scores))

    if geom_mean >= 0.8:
        return 'A'
    elif geom_mean >= 0.6:
        return 'B'
    elif geom_mean >= 0.4:
        return 'C'
    elif geom_mean >= 0.2:
        return 'D'
    else:
        return 'F'
