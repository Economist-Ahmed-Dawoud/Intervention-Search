"""
Time Series Intervention Visualization

Enables visualization of:
- "What if we intervened N days ago?" compared to actuals
- Counterfactual analysis on historical data
- Intervention impact over time

This is crucial for understanding:
1. How quickly interventions take effect
2. Whether interventions would have worked in the past
3. Optimal timing for interventions
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TimeSeriesInterventionResult:
    """Results from time series intervention analysis"""
    # Time series data
    dates: List[datetime]
    actual_values: pd.DataFrame  # Actual historical values
    counterfactual_values: pd.DataFrame  # What would have happened with intervention

    # Intervention details
    intervention_date: datetime
    intervention_spec: Dict[str, float]
    outcome_node: str

    # Summary statistics
    actual_outcome_series: pd.Series
    counterfactual_outcome_series: pd.Series
    causal_effect_series: pd.Series  # Difference over time
    cumulative_effect: float  # Total cumulative impact


class TimeSeriesInterventionAnalyzer:
    """
    Analyzes interventions on time series data.

    Answers questions like:
    - "What if we had reduced X by 10% starting 30 days ago?"
    - "How would the intervention propagate over time?"
    - "What's the cumulative impact?"
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
        Initialize time series analyzer.

        Args:
            graph: Causal DAG
            regressors_dict: Trained models
            baseline_stats: Baseline statistics
            node_types: Node types (categorical/continuous)
            label_encoders: Label encoders for categorical variables
        """
        self.graph = graph
        self.regressors_dict = regressors_dict
        self.baseline_stats = baseline_stats
        self.node_types = node_types
        self.label_encoders = label_encoders or {}

        # Import DO operator
        from ..core.do_operator import DOOperator

        self.do_operator = DOOperator(
            graph=graph,
            regressors_dict=regressors_dict,
            baseline_stats=baseline_stats,
            node_types=node_types,
            label_encoders=label_encoders
        )

        # Compute topological order
        try:
            self.topological_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            raise ValueError("Graph contains cycles!")

    def simulate_historical_intervention(
        self,
        historical_data: pd.DataFrame,
        intervention_node: str,
        intervention_pct_change: float,
        outcome_node: str,
        intervention_start_date: Union[str, datetime, int],
        date_column: Optional[str] = None
    ) -> TimeSeriesInterventionResult:
        """
        Simulate what would have happened if we intervened on a past date.

        Args:
            historical_data: DataFrame with historical observations
            intervention_node: Node to intervene on
            intervention_pct_change: Percentage change for intervention (e.g., -10 for 10% decrease)
            outcome_node: Node to track over time
            intervention_start_date: When intervention starts (date, string, or row index)
            date_column: Name of date column (if None, uses index)

        Returns:
            TimeSeriesInterventionResult with actual vs counterfactual comparison
        """
        # Prepare data
        if date_column is not None:
            if date_column not in historical_data.columns:
                raise ValueError(f"Date column '{date_column}' not found!")

            dates = pd.to_datetime(historical_data[date_column]).tolist()
            data = historical_data.copy()
        else:
            # Use index as dates
            if isinstance(historical_data.index, pd.DatetimeIndex):
                dates = historical_data.index.tolist()
            else:
                # Generate synthetic dates
                dates = [datetime.now() - timedelta(days=len(historical_data) - i - 1) for i in range(len(historical_data))]

            data = historical_data.copy()

        # Find intervention start index
        if isinstance(intervention_start_date, int):
            intervention_idx = intervention_start_date
        elif isinstance(intervention_start_date, (str, datetime)):
            intervention_date_obj = pd.to_datetime(intervention_start_date)
            # Find closest date
            date_diffs = [abs((d - intervention_date_obj).total_seconds()) for d in dates]
            intervention_idx = int(np.argmin(date_diffs))
        else:
            raise ValueError("intervention_start_date must be int, str, or datetime")

        intervention_date = dates[intervention_idx]

        # Validate nodes
        if intervention_node not in self.graph.nodes():
            raise ValueError(f"Intervention node '{intervention_node}' not in graph!")
        if outcome_node not in self.graph.nodes():
            raise ValueError(f"Outcome node '{outcome_node}' not in graph!")

        # Check for causal path
        if not nx.has_path(self.graph, intervention_node, outcome_node):
            warnings.warn(f"No causal path from {intervention_node} to {outcome_node}!")

        # Initialize result storage
        actual_values_list = []
        counterfactual_values_list = []

        # Simulate time series with intervention
        for t in range(len(data)):
            # Get actual values at time t
            actual_vals = {}
            for node in self.graph.nodes():
                if node in data.columns:
                    actual_vals[node] = data.iloc[t][node]
                else:
                    actual_vals[node] = self.baseline_stats.get(node, {}).get('mean', 0)

            actual_values_list.append(actual_vals.copy())

            # Compute counterfactual
            if t < intervention_idx:
                # Before intervention: counterfactual = actual
                counterfactual_vals = actual_vals.copy()
            else:
                # After intervention: apply DO operator
                baseline_val = actual_vals[intervention_node]
                intervention_val = baseline_val * (1 + intervention_pct_change / 100)

                # Use DO operator to compute counterfactual
                do_result = self.do_operator.do(
                    intervention_values={intervention_node: intervention_val},
                    baseline_values=actual_vals
                )

                counterfactual_vals = do_result.post_intervention_values

            counterfactual_values_list.append(counterfactual_vals.copy())

        # Convert to DataFrames
        actual_df = pd.DataFrame(actual_values_list, index=dates)
        counterfactual_df = pd.DataFrame(counterfactual_values_list, index=dates)

        # Extract outcome series
        actual_outcome_series = actual_df[outcome_node]
        counterfactual_outcome_series = counterfactual_df[outcome_node]
        causal_effect_series = counterfactual_outcome_series - actual_outcome_series

        # Compute cumulative effect (after intervention)
        cumulative_effect = causal_effect_series.iloc[intervention_idx:].sum()

        # Create intervention spec
        baseline_val = actual_df.iloc[intervention_idx][intervention_node]
        intervention_val = baseline_val * (1 + intervention_pct_change / 100)
        intervention_spec = {intervention_node: intervention_val}

        return TimeSeriesInterventionResult(
            dates=dates,
            actual_values=actual_df,
            counterfactual_values=counterfactual_df,
            intervention_date=intervention_date,
            intervention_spec=intervention_spec,
            outcome_node=outcome_node,
            actual_outcome_series=actual_outcome_series,
            counterfactual_outcome_series=counterfactual_outcome_series,
            causal_effect_series=causal_effect_series,
            cumulative_effect=cumulative_effect
        )

    def plot_intervention_comparison(
        self,
        result: TimeSeriesInterventionResult,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        show_intervention_line: bool = True
    ):
        """
        Plot actual vs counterfactual comparison.

        Args:
            result: TimeSeriesInterventionResult
            title: Plot title
            figsize: Figure size
            show_intervention_line: Whether to show vertical line at intervention date

        Returns:
            matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot 1: Actual vs Counterfactual
        ax1 = axes[0]
        ax1.plot(result.dates, result.actual_outcome_series, label='Actual', color='blue', linewidth=2)
        ax1.plot(result.dates, result.counterfactual_outcome_series, label='Counterfactual (with intervention)',
                 color='green', linestyle='--', linewidth=2)

        if show_intervention_line:
            ax1.axvline(x=result.intervention_date, color='red', linestyle=':', linewidth=2,
                       label='Intervention Start', alpha=0.7)

        ax1.set_ylabel(f'{result.outcome_node}', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        if title:
            ax1.set_title(title, fontsize=14, fontweight='bold')
        else:
            intervention_node = list(result.intervention_spec.keys())[0]
            ax1.set_title(
                f'Intervention Impact: {intervention_node} → {result.outcome_node}',
                fontsize=14, fontweight='bold'
            )

        # Plot 2: Causal Effect Over Time
        ax2 = axes[1]
        ax2.plot(result.dates, result.causal_effect_series, label='Causal Effect', color='purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        if show_intervention_line:
            ax2.axvline(x=result.intervention_date, color='red', linestyle=':', linewidth=2, alpha=0.7)

        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Causal Effect', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add summary text
        cumulative_text = f'Cumulative Effect: {result.cumulative_effect:.2f}'
        avg_effect = result.causal_effect_series.iloc[
            result.dates.index(result.intervention_date):
        ].mean() if result.intervention_date in result.dates else 0
        avg_text = f'Avg Effect: {avg_effect:.2f}'

        ax2.text(
            0.02, 0.98, f'{cumulative_text}\n{avg_text}',
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        return fig, axes

    def analyze_optimal_intervention_timing(
        self,
        historical_data: pd.DataFrame,
        intervention_node: str,
        intervention_pct_change: float,
        outcome_node: str,
        date_column: Optional[str] = None,
        metric: str = 'cumulative_effect'
    ) -> pd.DataFrame:
        """
        Analyze when would have been the best time to intervene.

        Simulates intervention at each possible date and ranks by effectiveness.

        Args:
            historical_data: Historical data
            intervention_node: Node to intervene on
            intervention_pct_change: Intervention magnitude
            outcome_node: Outcome to measure
            date_column: Date column name
            metric: Metric to optimize ('cumulative_effect', 'avg_effect', 'max_effect')

        Returns:
            DataFrame with intervention effectiveness by start date
        """
        results = []

        # Try intervention at each possible date
        n_dates = len(historical_data)

        for start_idx in range(0, n_dates - 1):  # Leave at least 1 day for effect
            try:
                result = self.simulate_historical_intervention(
                    historical_data=historical_data,
                    intervention_node=intervention_node,
                    intervention_pct_change=intervention_pct_change,
                    outcome_node=outcome_node,
                    intervention_start_date=start_idx,
                    date_column=date_column
                )

                # Compute metrics
                post_intervention_effects = result.causal_effect_series.iloc[start_idx:]
                cumulative = result.cumulative_effect
                avg_effect = post_intervention_effects.mean()
                max_effect = post_intervention_effects.max()
                min_effect = post_intervention_effects.min()

                results.append({
                    'intervention_start_date': result.intervention_date,
                    'intervention_start_idx': start_idx,
                    'cumulative_effect': cumulative,
                    'avg_effect': avg_effect,
                    'max_effect': max_effect,
                    'min_effect': min_effect,
                    'n_days_active': len(post_intervention_effects)
                })

            except Exception as e:
                continue

        results_df = pd.DataFrame(results)

        # Sort by specified metric (descending for positive effects)
        if not results_df.empty:
            results_df = results_df.sort_values(by=metric, ascending=False)

        return results_df

    def export_counterfactual_data(
        self,
        result: TimeSeriesInterventionResult,
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export counterfactual analysis results to file.

        Args:
            result: TimeSeriesInterventionResult
            output_path: Output file path
            format: 'csv', 'excel', or 'json'
        """
        # Combine data
        export_df = pd.DataFrame({
            'date': result.dates,
            f'{result.outcome_node}_actual': result.actual_outcome_series,
            f'{result.outcome_node}_counterfactual': result.counterfactual_outcome_series,
            f'{result.outcome_node}_causal_effect': result.causal_effect_series
        })

        # Add intervention marker
        intervention_idx = result.dates.index(result.intervention_date)
        export_df['intervention_active'] = False
        export_df.loc[intervention_idx:, 'intervention_active'] = True

        # Export
        if format == 'csv':
            export_df.to_csv(output_path, index=False)
        elif format == 'excel':
            export_df.to_excel(output_path, index=False)
        elif format == 'json':
            export_df.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Exported counterfactual data to: {output_path}")


def create_intervention_report(
    result: TimeSeriesInterventionResult,
    include_summary_stats: bool = True
) -> str:
    """
    Generate a text report summarizing the intervention analysis.

    Args:
        result: TimeSeriesInterventionResult
        include_summary_stats: Whether to include detailed statistics

    Returns:
        Formatted text report
    """
    intervention_node = list(result.intervention_spec.keys())[0]
    intervention_val = result.intervention_spec[intervention_node]

    intervention_idx = result.dates.index(result.intervention_date)
    baseline_val = result.actual_values.iloc[intervention_idx][intervention_node]
    pct_change = ((intervention_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0

    report = []
    report.append("=" * 70)
    report.append("📊 COUNTERFACTUAL INTERVENTION ANALYSIS")
    report.append("=" * 70)
    report.append("")
    report.append(f"Intervention Node: {intervention_node}")
    report.append(f"Intervention Date: {result.intervention_date.strftime('%Y-%m-%d')}")
    report.append(f"Intervention: {baseline_val:.2f} → {intervention_val:.2f} ({pct_change:+.1f}%)")
    report.append(f"Outcome Node: {result.outcome_node}")
    report.append("")
    report.append("─" * 70)
    report.append("IMPACT SUMMARY")
    report.append("─" * 70)
    report.append(f"Cumulative Effect: {result.cumulative_effect:+.2f}")

    post_intervention_effects = result.causal_effect_series.iloc[intervention_idx:]
    if len(post_intervention_effects) > 0:
        report.append(f"Average Effect: {post_intervention_effects.mean():+.2f}")
        report.append(f"Max Effect: {post_intervention_effects.max():+.2f}")
        report.append(f"Min Effect: {post_intervention_effects.min():+.2f}")
        report.append(f"Days Active: {len(post_intervention_effects)}")

    if include_summary_stats:
        report.append("")
        report.append("─" * 70)
        report.append("OUTCOME STATISTICS")
        report.append("─" * 70)
        report.append(f"Actual Outcome (mean): {result.actual_outcome_series.mean():.2f}")
        report.append(f"Counterfactual Outcome (mean): {result.counterfactual_outcome_series.mean():.2f}")
        report.append(f"Absolute Change: {(result.counterfactual_outcome_series.mean() - result.actual_outcome_series.mean()):+.2f}")

        if result.actual_outcome_series.mean() != 0:
            pct_improvement = (
                (result.counterfactual_outcome_series.mean() - result.actual_outcome_series.mean()) /
                result.actual_outcome_series.mean()
            ) * 100
            report.append(f"Percentage Change: {pct_improvement:+.1f}%")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)
