"""
Intervention Ranker

Multi-objective ranking of intervention recommendations based on:
1. Effect size (how close to target)
2. Uncertainty (how confident)
3. Feasibility (how practical)
4. Model quality (how reliable)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RankingWeights:
    """Weights for ranking criteria"""
    accuracy: float = 0.35  # How close to target
    uncertainty: float = 0.25  # Low uncertainty is better
    model_quality: float = 0.25  # High quality models
    simplicity: float = 0.15  # Fewer nodes is simpler

    def __post_init__(self):
        # Normalize weights to sum to 1
        total = self.accuracy + self.uncertainty + self.model_quality + self.simplicity
        self.accuracy /= total
        self.uncertainty /= total
        self.model_quality /= total
        self.simplicity /= total


class InterventionRanker:
    """
    Ranks intervention recommendations using multi-objective scoring.
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        prioritize_safety: bool = True
    ):
        """
        Initialize ranker.

        Args:
            weights: Custom ranking weights
            prioritize_safety: If True, heavily penalize unsafe/infeasible interventions
        """
        self.weights = weights or RankingWeights()
        self.prioritize_safety = prioritize_safety

    def rank_interventions(
        self,
        interventions: List[Dict],
        target_effect: float,
        tolerance: float
    ) -> List[Dict]:
        """
        Rank interventions by multi-objective score.

        Args:
            interventions: List of intervention dictionaries
            target_effect: Target effect size
            tolerance: Acceptable error

        Returns:
            Sorted list of interventions with scores added
        """
        if not interventions:
            return []

        # Compute scores for each intervention
        for intervention in interventions:
            scores = self._compute_scores(intervention, target_effect, tolerance)
            intervention['ranking_scores'] = scores
            intervention['overall_score'] = scores['overall']

        # Sort by overall score (descending)
        sorted_interventions = sorted(
            interventions,
            key=lambda x: x['overall_score'],
            reverse=True
        )

        # Add rank
        for rank, intervention in enumerate(sorted_interventions, 1):
            intervention['rank'] = rank

        return sorted_interventions

    def _compute_scores(
        self,
        intervention: Dict,
        target_effect: float,
        tolerance: float
    ) -> Dict:
        """
        Compute individual component scores.

        Args:
            intervention: Intervention dict
            target_effect: Target effect
            tolerance: Acceptable error

        Returns:
            Dictionary of scores
        """
        # Score 1: Accuracy (how close to target)
        predicted_effect = intervention.get('actual_effect', 0)
        error = abs(predicted_effect - target_effect)

        if abs(target_effect) > 1e-9:
            error_pct = error / abs(target_effect)
            accuracy_score = np.exp(-error_pct)  # Exponential decay
        else:
            accuracy_score = 1.0 if error < tolerance else 0.5

        # Bonus for being within tolerance
        if error <= tolerance:
            accuracy_score = min(1.0, accuracy_score * 1.2)

        # Score 2: Uncertainty (lower is better)
        uncertainty_std = intervention.get('prediction_uncertainty_std', 0)

        if abs(predicted_effect) > 1e-9:
            cv = uncertainty_std / abs(predicted_effect)
            uncertainty_score = np.exp(-cv)
        else:
            uncertainty_score = 0.5

        # Check for confidence interval if available
        if 'prediction_interval_lower' in intervention and 'prediction_interval_upper' in intervention:
            ci_lower = intervention['prediction_interval_lower']
            ci_upper = intervention['prediction_interval_upper']
            ci_width = abs(ci_upper - ci_lower)

            # Penalize wide confidence intervals
            if abs(predicted_effect) > 1e-9:
                ci_width_relative = ci_width / abs(predicted_effect)
                ci_penalty = min(1.0, ci_width_relative / 2.0)
                uncertainty_score *= (1 - ci_penalty * 0.5)

        # Score 3: Model quality
        quality = intervention.get('quality', {})
        if isinstance(quality, dict):
            model_quality_score = quality.get('quality_score_geom_mean', 0.5)
        else:
            # Fallback to confidence if available
            model_quality_score = intervention.get('confidence', 0.5)

        # Score 4: Simplicity (prefer fewer intervention nodes)
        nodes = intervention.get('nodes', [])
        n_nodes = len(nodes)
        simplicity_score = 1.0 / (1.0 + 0.5 * (n_nodes - 1))  # Penalize multi-node

        # Safety penalty
        safety_multiplier = 1.0
        if self.prioritize_safety:
            validation = intervention.get('validation', {})

            if not validation.get('is_feasible', True):
                safety_multiplier *= 0.3  # Heavy penalty for infeasible

            if not validation.get('is_safe', True):
                safety_multiplier *= 0.7  # Penalty for OOD

            # Apply confidence adjustment from validation
            conf_adjustment = validation.get('confidence_adjustment', 1.0)
            safety_multiplier *= conf_adjustment

        # Combine scores
        overall_score = (
            self.weights.accuracy * accuracy_score +
            self.weights.uncertainty * uncertainty_score +
            self.weights.model_quality * model_quality_score +
            self.weights.simplicity * simplicity_score
        ) * safety_multiplier

        return {
            'overall': np.clip(overall_score, 0.0, 1.0),
            'accuracy': accuracy_score,
            'uncertainty': uncertainty_score,
            'model_quality': model_quality_score,
            'simplicity': simplicity_score,
            'safety_multiplier': safety_multiplier
        }

    def get_top_k(
        self,
        interventions: List[Dict],
        k: int,
        target_effect: float,
        tolerance: float,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Get top K interventions.

        Args:
            interventions: List of interventions
            k: Number to return
            target_effect: Target effect
            tolerance: Error tolerance
            min_score: Minimum score threshold

        Returns:
            Top K interventions
        """
        ranked = self.rank_interventions(interventions, target_effect, tolerance)

        # Filter by minimum score
        filtered = [i for i in ranked if i['overall_score'] >= min_score]

        return filtered[:k]

    def create_comparison_table(
        self,
        interventions: List[Dict],
        target_effect: float,
        tolerance: float
    ) -> pd.DataFrame:
        """
        Create a comparison table for interventions.

        Args:
            interventions: List of interventions
            target_effect: Target effect
            tolerance: Error tolerance

        Returns:
            Pandas DataFrame with comparison
        """
        ranked = self.rank_interventions(interventions, target_effect, tolerance)

        rows = []
        for intervention in ranked[:10]:  # Top 10
            scores = intervention.get('ranking_scores', {})

            row = {
                'Rank': intervention.get('rank', 0),
                'Type': intervention.get('intervention_type', 'unknown'),
                'Variables': ', '.join(intervention.get('nodes', [])),
                'Effect': f"{intervention.get('actual_effect', 0):+.1f}%",
                'Error': f"{intervention.get('error_from_target', 0):.1f}%",
                'Overall_Score': f"{scores.get('overall', 0):.2f}",
                'Accuracy': f"{scores.get('accuracy', 0):.2f}",
                'Confidence': f"{intervention.get('confidence', 0):.0%}",
                'Within_Tol': '✅' if intervention.get('within_tolerance', False) else '❌',
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def explain_ranking(
        self,
        intervention: Dict,
        target_effect: float,
        tolerance: float
    ) -> str:
        """
        Generate human-readable explanation of ranking.

        Args:
            intervention: Intervention dict
            target_effect: Target effect
            tolerance: Tolerance

        Returns:
            Explanation string
        """
        scores = intervention.get('ranking_scores', self._compute_scores(intervention, target_effect, tolerance))

        rank = intervention.get('rank', '?')
        overall = scores['overall']

        explanation_parts = [
            f"Rank #{rank} (Overall Score: {overall:.2f})",
            "",
            "Component Scores:",
            f"  • Accuracy: {scores['accuracy']:.2f} - " + self._explain_accuracy(intervention, target_effect),
            f"  • Uncertainty: {scores['uncertainty']:.2f} - " + self._explain_uncertainty(intervention),
            f"  • Model Quality: {scores['model_quality']:.2f} - " + self._explain_quality(intervention),
            f"  • Simplicity: {scores['simplicity']:.2f} - " + self._explain_simplicity(intervention),
        ]

        if scores['safety_multiplier'] < 1.0:
            explanation_parts.append("")
            explanation_parts.append(f"  ⚠️  Safety Penalty: {scores['safety_multiplier']:.2f}x")

        return "\n".join(explanation_parts)

    def _explain_accuracy(self, intervention: Dict, target: float) -> str:
        error = intervention.get('error_from_target', 0)
        if error <= 2.0:
            return f"Very close to target (error: {error:.1f}%)"
        elif error <= 5.0:
            return f"Close to target (error: {error:.1f}%)"
        else:
            return f"Far from target (error: {error:.1f}%)"

    def _explain_uncertainty(self, intervention: Dict) -> str:
        if 'prediction_interval_lower' in intervention and 'prediction_interval_upper' in intervention:
            ci_lower = intervention['prediction_interval_lower']
            ci_upper = intervention['prediction_interval_upper']
            width = abs(ci_upper - ci_lower)
            return f"90% CI width: {width:.1f}%"
        else:
            conf = intervention.get('confidence', 0.5)
            return f"Confidence: {conf:.0%}"

    def _explain_quality(self, intervention: Dict) -> str:
        quality = intervention.get('quality', {})
        if isinstance(quality, dict):
            grade = quality.get('quality_grade', 'Unknown')
            return f"Model grade: {grade}"
        else:
            return "Quality information not available"

    def _explain_simplicity(self, intervention: Dict) -> str:
        nodes = intervention.get('nodes', [])
        n = len(nodes)
        if n == 1:
            return "Single-node intervention (simple)"
        else:
            return f"{n}-node intervention (complex)"
