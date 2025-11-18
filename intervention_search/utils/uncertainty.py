"""
Uncertainty Quantification Utilities for Intervention Search

This module provides tools for properly estimating and propagating uncertainty
through causal graphs during intervention simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates"""
    mean: float
    std: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    samples: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'mean': float(self.mean),
            'std': float(self.std),
            'ci_90': [float(self.percentile_5), float(self.percentile_95)],
            'ci_50': [float(self.percentile_25), float(self.percentile_75)],
            'median': float(self.percentile_50)
        }


def estimate_uncertainty_from_samples(samples: np.ndarray) -> UncertaintyEstimate:
    """
    Calculate comprehensive uncertainty metrics from Monte Carlo samples.

    Uses linear interpolation for more accurate percentile estimates.

    Args:
        samples: Array of Monte Carlo samples

    Returns:
        UncertaintyEstimate object with all statistics
    """
    return UncertaintyEstimate(
        mean=np.mean(samples),
        std=np.std(samples, ddof=1),  # Unbiased estimator
        percentile_5=np.percentile(samples, 5, method='linear'),
        percentile_25=np.percentile(samples, 25, method='linear'),
        percentile_50=np.percentile(samples, 50, method='linear'),
        percentile_75=np.percentile(samples, 75, method='linear'),
        percentile_95=np.percentile(samples, 95, method='linear'),
        samples=samples
    )


def compute_prediction_interval(
    predictions: np.ndarray,
    confidence_level: float = 0.90
) -> Tuple[float, float]:
    """
    Compute prediction interval from Monte Carlo samples.

    Uses linear interpolation for more accurate percentile estimates.

    Args:
        predictions: Array of predicted values
        confidence_level: Desired confidence level (default: 0.90)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(predictions, lower_percentile, method='linear')
    upper = np.percentile(predictions, upper_percentile, method='linear')

    return lower, upper


def propagate_uncertainty_analytical(
    base_mean: float,
    base_std: float,
    model_rmse: float
) -> Tuple[float, float]:
    """
    Analytically propagate uncertainty through a single node.

    For a regression model: Y = f(X) + ε where ε ~ N(0, RMSE²)
    If X ~ N(μ_x, σ_x²), then (approximately):
    Y ~ N(f(μ_x), σ_x² * (df/dx)² + RMSE²)

    For linear case where df/dx ≈ 1:
    σ_y² = σ_x² + RMSE²

    Args:
        base_mean: Mean of input
        base_std: Std dev of input
        model_rmse: RMSE of the model

    Returns:
        Tuple of (propagated_mean, propagated_std)
    """
    # Assuming approximately linear propagation
    propagated_std = np.sqrt(base_std**2 + model_rmse**2)
    return base_mean, propagated_std


def estimate_model_uncertainty(
    predictions: List[float],
    ground_truth: Optional[List[float]] = None,
    ensemble_predictions: Optional[List[np.ndarray]] = None
) -> Dict:
    """
    Estimate model uncertainty from various sources.

    Combines:
    1. Aleatoric uncertainty (data noise)
    2. Epistemic uncertainty (model uncertainty)

    Args:
        predictions: Model predictions
        ground_truth: True values (if available)
        ensemble_predictions: Predictions from ensemble members

    Returns:
        Dictionary with uncertainty estimates
    """
    result = {}

    # Aleatoric uncertainty (irreducible)
    if ground_truth is not None:
        residuals = np.array(predictions) - np.array(ground_truth)
        result['aleatoric_std'] = np.std(residuals)
        result['rmse'] = np.sqrt(np.mean(residuals**2))

    # Epistemic uncertainty (model uncertainty, reducible with more data)
    if ensemble_predictions is not None:
        ensemble_array = np.array(ensemble_predictions)  # Shape: (n_models, n_samples)
        result['epistemic_std'] = np.std(ensemble_array, axis=0).mean()
        result['ensemble_mean'] = np.mean(ensemble_array, axis=0)
        result['ensemble_std'] = np.std(ensemble_array, axis=0)

    # Total uncertainty
    if 'aleatoric_std' in result and 'epistemic_std' in result:
        result['total_std'] = np.sqrt(
            result['aleatoric_std']**2 + result['epistemic_std']**2
        )

    return result


def compute_intervention_confidence(
    predicted_effect: float,
    target_effect: float,
    uncertainty_std: float,
    model_quality_scores: List[float]
) -> float:
    """
    Compute confidence score for an intervention recommendation.

    Combines:
    1. How close predicted effect is to target
    2. Uncertainty magnitude
    3. Quality of models in causal path

    Args:
        predicted_effect: Predicted effect size
        target_effect: Desired effect size
        uncertainty_std: Standard deviation of prediction
        model_quality_scores: List of R² scores for models in path

    Returns:
        Confidence score between 0 and 1
    """
    # Factor 1: Accuracy (how close to target)
    if abs(target_effect) > 1e-9:
        error_pct = abs(predicted_effect - target_effect) / abs(target_effect)
        accuracy_score = np.exp(-error_pct)  # Exponential decay
    else:
        accuracy_score = 0.5

    # Factor 2: Precision (low uncertainty)
    if abs(predicted_effect) > 1e-9:
        cv = uncertainty_std / abs(predicted_effect)  # Coefficient of variation
        precision_score = np.exp(-cv)  # Lower CV = higher precision
    else:
        precision_score = 0.5

    # Factor 3: Model quality (weakest link)
    if model_quality_scores:
        # Use geometric mean to penalize weak links heavily
        quality_score = np.prod(model_quality_scores) ** (1.0 / len(model_quality_scores))
    else:
        quality_score = 0.5

    # Weighted combination
    confidence = (
        0.4 * accuracy_score +
        0.3 * precision_score +
        0.3 * quality_score
    )

    return np.clip(confidence, 0.0, 1.0)


def estimate_causal_effect_variance(
    path_length: int,
    node_rmses: List[float],
    node_stds: List[float]
) -> float:
    """
    Estimate variance of causal effect estimate through a path.

    For a causal path X → Y → Z, the variance compounds:
    Var(Z|do(X)) ≈ Var(Y|do(X)) * β²_YZ + σ²_Z

    Args:
        path_length: Number of edges in causal path
        node_rmses: RMSE values for each node's model
        node_stds: Standard deviations of each node in training data

    Returns:
        Estimated variance of the causal effect
    """
    # Start with intervention node (zero variance since we set it)
    accumulated_variance = 0.0

    # Propagate through each hop
    for i in range(path_length):
        rmse = node_rmses[i] if i < len(node_rmses) else 0.0
        # Assume effect attenuates by ~0.8 per hop (conservative)
        attenuation = 0.8 ** i
        accumulated_variance += (attenuation * rmse) ** 2

    return accumulated_variance


def compute_uncertainty_intervals(
    samples: np.ndarray,
    confidence_levels: List[float] = [0.68, 0.90, 0.95]
) -> Dict[float, Tuple[float, float]]:
    """
    Compute multiple confidence intervals from samples.

    Args:
        samples: Monte Carlo samples
        confidence_levels: List of confidence levels to compute

    Returns:
        Dictionary mapping confidence level to (lower, upper) bounds
    """
    intervals = {}

    for conf in confidence_levels:
        lower, upper = compute_prediction_interval(samples, conf)
        intervals[conf] = (lower, upper)

    return intervals
