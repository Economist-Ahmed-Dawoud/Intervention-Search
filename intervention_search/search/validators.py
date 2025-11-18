"""
Intervention Validators

Validates intervention recommendations for feasibility, out-of-distribution risks,
and practical constraints.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Result from intervention validation"""
    is_valid: bool
    is_feasible: bool
    is_safe: bool  # Not in OOD territory
    confidence_adjustment: float  # Multiplier to adjust confidence (0-1)
    warnings: List[str]
    errors: List[str]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class OutOfDistributionDetector:
    """
    Detects when intervention pushes variables outside their training distribution.

    Critical for preventing recommendations that look good on paper but fail in practice.
    """

    def __init__(
        self,
        baseline_stats: Dict[str, Dict],
        strict_mode: bool = False,
        ood_threshold_std: float = 2.5
    ):
        """
        Initialize OOD detector.

        Args:
            baseline_stats: Training data statistics (mean, std, min, max)
            strict_mode: If True, apply stricter bounds
            ood_threshold_std: Number of std devs before flagging OOD
        """
        self.baseline_stats = baseline_stats
        self.strict_mode = strict_mode
        self.ood_threshold_std = ood_threshold_std

        if strict_mode:
            self.ood_threshold_std = 2.0

    def check_value_in_distribution(
        self,
        node: str,
        value: float
    ) -> Dict:
        """
        Check if a value is within the training distribution for a node.

        Args:
            node: Node name
            value: Proposed value

        Returns:
            Dictionary with OOD assessment
        """
        if node not in self.baseline_stats:
            return {
                'in_distribution': True,
                'warning': None,
                'distance_from_mean_std': 0,
                'reason': 'No baseline stats available'
            }

        stats = self.baseline_stats[node]
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        min_val = stats.get('min', float('-inf'))
        max_val = stats.get('max', float('inf'))

        # Check 1: Outside [min, max] range
        if value < min_val or value > max_val:
            return {
                'in_distribution': False,
                'warning': f"Value {value:.2f} outside training range [{min_val:.2f}, {max_val:.2f}]",
                'distance_from_mean_std': abs(value - mean) / std if std > 0 else 0,
                'severity': 'high'
            }

        # Check 2: Outside [mean - k*std, mean + k*std]
        if std > 0:
            distance_std = abs(value - mean) / std

            if distance_std > self.ood_threshold_std:
                return {
                    'in_distribution': False,
                    'warning': f"Value {value:.2f} is {distance_std:.1f} std devs from mean {mean:.2f}",
                    'distance_from_mean_std': distance_std,
                    'severity': 'medium' if distance_std < 3.0 else 'high'
                }

        # Value is in distribution
        return {
            'in_distribution': True,
            'warning': None,
            'distance_from_mean_std': abs(value - mean) / std if std > 0 else 0,
            'severity': 'none'
        }

    def validate_intervention(
        self,
        intervention_dict: Dict[str, float]
    ) -> Dict:
        """
        Validate all intervention values are in distribution.

        Args:
            intervention_dict: Dictionary of {node: intervention_value}

        Returns:
            Validation results
        """
        ood_nodes = []
        warnings_list = []
        max_severity = 'none'

        for node, value in intervention_dict.items():
            check = self.check_value_in_distribution(node, value)

            if not check['in_distribution']:
                ood_nodes.append({
                    'node': node,
                    'value': value,
                    'warning': check['warning'],
                    'severity': check['severity'],
                    'distance_std': check['distance_from_mean_std']
                })
                warnings_list.append(check['warning'])

                # Track max severity
                if check['severity'] == 'high':
                    max_severity = 'high'
                elif check['severity'] == 'medium' and max_severity != 'high':
                    max_severity = 'medium'

        is_safe = len(ood_nodes) == 0

        # Compute confidence adjustment
        if max_severity == 'high':
            confidence_adjustment = 0.5  # Cut confidence in half
        elif max_severity == 'medium':
            confidence_adjustment = 0.75  # Reduce confidence by 25%
        else:
            confidence_adjustment = 1.0  # No adjustment

        return {
            'is_safe': is_safe,
            'ood_nodes': ood_nodes,
            'warnings': warnings_list,
            'max_severity': max_severity,
            'confidence_adjustment': confidence_adjustment
        }

    def validate_propagated_values(
        self,
        propagated_values: Dict[str, float]
    ) -> Dict:
        """
        Check if propagated values end up in OOD territory.

        Args:
            propagated_values: Dictionary of propagated node values

        Returns:
            Validation results
        """
        return self.validate_intervention(propagated_values)


class FeasibilityChecker:
    """
    Checks if interventions are practically feasible.
    """

    def __init__(
        self,
        constraints: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize feasibility checker.

        Args:
            constraints: Dictionary of constraints per node
                Example: {
                    'cpu_usage': {'min': 0, 'max': 100, 'step': 1},
                    'staffing': {'min': 10, 'max': 50, 'integer': True}
                }
        """
        self.constraints = constraints or {}

    def check_constraint_satisfaction(
        self,
        node: str,
        value: float
    ) -> Dict:
        """
        Check if a value satisfies constraints.

        Args:
            node: Node name
            value: Proposed value

        Returns:
            Dictionary with constraint check results
        """
        if node not in self.constraints:
            return {
                'satisfies_constraints': True,
                'violations': []
            }

        constraints = self.constraints[node]
        violations = []

        # Check min/max bounds
        if 'min' in constraints and value < constraints['min']:
            violations.append(f"Value {value:.2f} below minimum {constraints['min']}")

        if 'max' in constraints and value > constraints['max']:
            violations.append(f"Value {value:.2f} above maximum {constraints['max']}")

        # Check integer constraint
        if constraints.get('integer', False) and not np.isclose(value, round(value)):
            violations.append(f"Value {value:.2f} must be integer")

        # Check step constraint
        if 'step' in constraints:
            step = constraints['step']
            min_val = constraints.get('min', 0)
            steps_from_min = (value - min_val) / step
            if not np.isclose(steps_from_min, round(steps_from_min)):
                violations.append(f"Value {value:.2f} must be multiple of step {step}")

        return {
            'satisfies_constraints': len(violations) == 0,
            'violations': violations
        }

    def validate_intervention(
        self,
        intervention_dict: Dict[str, float]
    ) -> Dict:
        """
        Validate entire intervention against constraints.

        Args:
            intervention_dict: Intervention specification

        Returns:
            Validation results
        """
        all_violations = []

        for node, value in intervention_dict.items():
            check = self.check_constraint_satisfaction(node, value)
            if not check['satisfies_constraints']:
                for violation in check['violations']:
                    all_violations.append(f"{node}: {violation}")

        return {
            'is_feasible': len(all_violations) == 0,
            'violations': all_violations
        }


class InterventionValidator:
    """
    Combined validation of intervention recommendations.

    Integrates:
    1. Out-of-distribution detection
    2. Feasibility checking
    3. Sanity checks
    """

    def __init__(
        self,
        baseline_stats: Dict[str, Dict],
        constraints: Optional[Dict[str, Dict]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize validator.

        Args:
            baseline_stats: Training data statistics
            constraints: Feasibility constraints
            strict_mode: Apply strict validation
        """
        self.ood_detector = OutOfDistributionDetector(baseline_stats, strict_mode)
        self.feasibility_checker = FeasibilityChecker(constraints)
        self.strict_mode = strict_mode

    def validate_intervention_recommendation(
        self,
        intervention_dict: Dict[str, float],
        predicted_outcome: float,
        target_outcome: float,
        propagated_values: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of an intervention recommendation.

        Args:
            intervention_dict: Proposed intervention
            predicted_outcome: Predicted effect
            target_outcome: Target effect
            propagated_values: All propagated node values (optional)

        Returns:
            ValidationResult
        """
        warnings_list = []
        errors_list = []

        # Check 1: OOD detection for intervention values
        ood_check = self.ood_detector.validate_intervention(intervention_dict)
        if not ood_check['is_safe']:
            warnings_list.extend(ood_check['warnings'])

        # Check 2: OOD detection for propagated values
        if propagated_values is not None:
            prop_check = self.ood_detector.validate_propagated_values(propagated_values)
            if not prop_check['is_safe']:
                # Filter to only severe OOD cases in propagated values
                severe_ood = [n for n in prop_check['ood_nodes'] if n['severity'] == 'high']
                if severe_ood:
                    warnings_list.append(
                        f"Intervention causes {len(severe_ood)} node(s) to go severely out-of-distribution"
                    )

        # Check 3: Feasibility
        feasibility_check = self.feasibility_checker.validate_intervention(intervention_dict)
        if not feasibility_check['is_feasible']:
            errors_list.extend(feasibility_check['violations'])

        # Check 4: Sanity checks
        sanity_warnings = self._sanity_checks(
            intervention_dict, predicted_outcome, target_outcome
        )
        warnings_list.extend(sanity_warnings)

        # Determine validity
        is_valid = len(errors_list) == 0
        is_feasible = feasibility_check['is_feasible']
        is_safe = ood_check['is_safe']

        # Compute confidence adjustment
        confidence_adjustment = ood_check['confidence_adjustment']
        if not is_safe:
            confidence_adjustment *= 0.8

        return ValidationResult(
            is_valid=is_valid,
            is_feasible=is_feasible,
            is_safe=is_safe,
            confidence_adjustment=confidence_adjustment,
            warnings=warnings_list,
            errors=errors_list
        )

    def _sanity_checks(
        self,
        intervention_dict: Dict[str, float],
        predicted_outcome: float,
        target_outcome: float
    ) -> List[str]:
        """
        Run sanity checks on intervention.

        Args:
            intervention_dict: Intervention
            predicted_outcome: Predicted effect
            target_outcome: Target effect

        Returns:
            List of warning messages
        """
        warnings_list = []

        # Check 1: Intervention is non-zero
        if all(abs(v) < 1e-9 for v in intervention_dict.values()):
            warnings_list.append("Intervention values are near zero")

        # Check 2: Effect is non-zero
        if abs(predicted_outcome) < 1e-9:
            warnings_list.append("Predicted effect is near zero")

        # Check 3: Effect is in right direction
        if abs(target_outcome) > 1e-9:
            same_direction = (predicted_outcome * target_outcome) > 0
            if not same_direction:
                warnings_list.append(
                    f"Predicted effect ({predicted_outcome:+.1f}) is opposite direction "
                    f"from target ({target_outcome:+.1f})"
                )

        # Check 4: Effect magnitude is reasonable
        if abs(target_outcome) > 1e-9:
            ratio = abs(predicted_outcome / target_outcome)
            if ratio > 5.0:
                warnings_list.append(
                    f"Predicted effect is {ratio:.1f}x larger than target - may be overestimated"
                )

        return warnings_list

    def filter_valid_interventions(
        self,
        interventions: List[Dict],
        require_feasible: bool = True,
        require_safe: bool = False
    ) -> List[Dict]:
        """
        Filter list of interventions to only valid ones.

        Args:
            interventions: List of intervention dictionaries
            require_feasible: Must satisfy constraints
            require_safe: Must be in-distribution

        Returns:
            Filtered list with validation info added
        """
        filtered = []

        for intervention in interventions:
            intervention_dict = intervention.get('required_pct_changes', {})
            baseline_values = {node: 0 for node in intervention_dict.keys()}  # Placeholder

            # Convert pct changes to absolute values (simplified)
            intervention_values = intervention_dict

            validation = self.validate_intervention_recommendation(
                intervention_values,
                intervention.get('actual_effect', 0),
                intervention.get('target_change', 0)
            )

            # Add validation info
            intervention['validation'] = {
                'is_valid': validation.is_valid,
                'is_feasible': validation.is_feasible,
                'is_safe': validation.is_safe,
                'confidence_adjustment': validation.confidence_adjustment,
                'warnings': validation.warnings,
                'errors': validation.errors
            }

            # Apply filters
            if require_feasible and not validation.is_feasible:
                continue
            if require_safe and not validation.is_safe:
                continue
            if not validation.is_valid:
                continue

            # Adjust confidence
            if 'confidence' in intervention:
                intervention['confidence'] *= validation.confidence_adjustment

            filtered.append(intervention)

        return filtered
