"""
Intervention Search - Production-Grade Causal Intervention System

A robust system for finding optimal causal interventions with proper uncertainty
quantification, model quality gating, and path sensitivity analysis.

Main Components:
    - InterventionSearch: Main public API
    - MonteCarloPropagator: Uncertainty-aware propagation
    - QualityGate: Model quality assessment
    - PathSensitivityAnalyzer: Causal path analysis
    - InterventionValidator: OOD detection and feasibility checking

Quick Start:
    >>> from intervention_search import InterventionSearch
    >>> from ht_categ import HT, HTConfig
    >>>
    >>> # Train your causal model
    >>> config = HTConfig(graph=adjacency_matrix, model_type='XGBoost')
    >>> ht_model = HT(config)
    >>> ht_model.train(training_data)
    >>>
    >>> # Search for interventions
    >>> searcher = InterventionSearch(ht_model.graph, ht_model)
    >>> results = searcher.find_interventions(
    ...     target_outcome='revenue',
    ...     target_change=+15,
    ...     tolerance=3.0
    ... )
    >>>
    >>> # Get recommendation
    >>> best = results['best_intervention']
    >>> print(f"Intervene on {best['nodes']}: {best['required_pct_changes']}")
    >>> print(f"Expected effect: {best['actual_effect']}% ± {best['prediction_uncertainty_std']}%")

Version: 2.0.0
"""

__version__ = "2.1.0"
__author__ = "Causal AI Team"

from .api.intervention_search import InterventionSearch
from .core.propagator import MonteCarloPropagator
from .core.quality_metrics import QualityGate, ModelQualityReport
from .core.path_analyzer import PathSensitivityAnalyzer
from .core.ensemble_trainer import EnsembleTrainer, integrate_ensemble_training_into_ht
from .core.do_operator import DOOperator, verify_do_operator_properties
from .search.validators import InterventionValidator, OutOfDistributionDetector
from .search.ranker import InterventionRanker, RankingWeights
from .search.optimizer import (
    AdaptiveGridSearch,
    BayesianOptimizer,
    MultiNodeOptimizer
)
from .visualization import (
    TimeSeriesInterventionAnalyzer,
    TimeSeriesInterventionResult,
    create_intervention_report
)

__all__ = [
    # Main API
    'InterventionSearch',

    # Core components
    'MonteCarloPropagator',
    'QualityGate',
    'ModelQualityReport',
    'PathSensitivityAnalyzer',
    'EnsembleTrainer',
    'integrate_ensemble_training_into_ht',
    'DOOperator',
    'verify_do_operator_properties',

    # Search components
    'InterventionValidator',
    'OutOfDistributionDetector',
    'InterventionRanker',
    'RankingWeights',

    # Optimizers
    'AdaptiveGridSearch',
    'BayesianOptimizer',
    'MultiNodeOptimizer',

    # Visualization
    'TimeSeriesInterventionAnalyzer',
    'TimeSeriesInterventionResult',
    'create_intervention_report',
]
