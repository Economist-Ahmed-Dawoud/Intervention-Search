"""
Comprehensive Demo of New Features in Intervention Search v2.1

This example demonstrates:
1. Ensemble training (multiple models per node)
2. DO operator methodology verification
3. Time series intervention visualization

New in v2.1:
- Train multiple models with varying complexity for each node
- Automatic best model selection
- Proper categorical variable handling
- 100% DO operator compliance verification
- Time series "what if" analysis
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

# Import main modules
from ht_categ import HT, HTConfig
from intervention_search import (
    InterventionSearch,
    EnsembleTrainer,
    integrate_ensemble_training_into_ht,
    DOOperator,
    verify_do_operator_properties,
    TimeSeriesInterventionAnalyzer,
    create_intervention_report
)


def generate_sample_data_with_categorical(n_samples=500, include_time_series=True):
    """
    Generate synthetic data with both continuous and categorical variables.

    Causal structure:
    - weather (categorical: sunny, rainy, cloudy) -> affects traffic and customer_mood
    - marketing_spend (continuous) -> customer_visits
    - traffic (continuous) -> customer_visits
    - customer_mood (categorical: happy, neutral, sad) -> conversion_rate
    - customer_visits (continuous) + conversion_rate (continuous) -> sales
    """
    np.random.seed(42)

    # Generate weather (categorical root)
    weather_categories = ['sunny', 'rainy', 'cloudy']
    weather = np.random.choice(weather_categories, n_samples, p=[0.5, 0.3, 0.2])

    # Generate marketing spend (continuous root)
    marketing_spend = np.random.uniform(1000, 5000, n_samples)

    # Generate traffic (affected by weather)
    weather_to_traffic = {'sunny': 100, 'rainy': 70, 'cloudy': 85}
    traffic = np.array([weather_to_traffic[w] for w in weather])
    traffic += np.random.normal(0, 10, n_samples)

    # Generate customer mood (affected by weather)
    customer_mood = []
    for w in weather:
        if w == 'sunny':
            mood = np.random.choice(['happy', 'neutral', 'sad'], p=[0.7, 0.2, 0.1])
        elif w == 'rainy':
            mood = np.random.choice(['happy', 'neutral', 'sad'], p=[0.2, 0.4, 0.4])
        else:  # cloudy
            mood = np.random.choice(['happy', 'neutral', 'sad'], p=[0.4, 0.4, 0.2])
        customer_mood.append(mood)
    customer_mood = np.array(customer_mood)

    # Generate customer visits (affected by marketing and traffic)
    customer_visits = 50 + 0.01 * marketing_spend + 0.5 * traffic + np.random.normal(0, 10, n_samples)

    # Generate conversion rate (affected by customer mood)
    mood_to_conversion = {'happy': 0.25, 'neutral': 0.15, 'sad': 0.08}
    conversion_rate = np.array([mood_to_conversion[m] for m in customer_mood])
    conversion_rate += np.random.normal(0, 0.02, n_samples)
    conversion_rate = np.clip(conversion_rate, 0.01, 0.5)

    # Generate sales (affected by visits and conversion)
    sales = customer_visits * conversion_rate * 100 + np.random.normal(0, 50, n_samples)
    sales = np.maximum(sales, 0)

    df = pd.DataFrame({
        'weather': weather,
        'marketing_spend': marketing_spend,
        'traffic': traffic,
        'customer_mood': customer_mood,
        'customer_visits': customer_visits,
        'conversion_rate': conversion_rate,
        'sales': sales
    })

    # Add time index if requested
    if include_time_series:
        start_date = datetime(2024, 1, 1)
        df['date'] = [start_date + timedelta(days=i) for i in range(n_samples)]

    return df


def create_sample_graph():
    """Create causal graph matching the data structure"""
    # Create adjacency matrix
    nodes = ['weather', 'marketing_spend', 'traffic', 'customer_mood',
             'customer_visits', 'conversion_rate', 'sales']

    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # Define edges (1 means edge exists)
    edges = [
        ('weather', 'traffic'),
        ('weather', 'customer_mood'),
        ('marketing_spend', 'customer_visits'),
        ('traffic', 'customer_visits'),
        ('customer_mood', 'conversion_rate'),
        ('customer_visits', 'sales'),
        ('conversion_rate', 'sales'),
    ]

    for src, dst in edges:
        adj_matrix.loc[src, dst] = 1

    return adj_matrix


def demo_ensemble_training():
    """Demonstrate ensemble training with multiple models per node"""
    print("\n" + "="*80)
    print("DEMO 1: ENSEMBLE TRAINING (Multiple Models Per Node)")
    print("="*80)

    # Generate data
    print("\n📊 Generating synthetic data...")
    df = generate_sample_data_with_categorical(n_samples=500, include_time_series=False)
    print(f"   Generated {len(df)} samples with {len(df.columns)} variables")
    print(f"   Categorical variables: weather, customer_mood")
    print(f"   Continuous variables: marketing_spend, traffic, customer_visits, conversion_rate, sales")

    # Create graph
    adj_matrix = create_sample_graph()

    # Train with ensemble
    print("\n🔧 Training HT model with ensemble training...")
    config = HTConfig(graph=adj_matrix, model_type='XGBoost')
    ht_model = HT(config)

    # Integrate ensemble training
    ht_model = integrate_ensemble_training_into_ht(
        ht_model,
        df,
        prefer_simpler_models=True,
        complexity_penalty=0.02
    )

    # Train with ensemble
    ht_model.train(df, use_ensemble=True)

    # Show model quality report
    print("\n📊 Model Quality Report:")
    ht_model.print_model_quality_report()

    # Show which models were selected
    if hasattr(ht_model, 'ensemble_selection_details'):
        print("\n🎯 Ensemble Model Selection Details:")
        for node, details in ht_model.ensemble_selection_details.items():
            print(f"   {node}: {details['model_name']}")
            if details.get('model_type') == 'regression':
                print(f"      → R²: {details.get('train_r2', 0):.3f} | CV: {details.get('cv_mean', 0):.3f}")
            else:
                print(f"      → Accuracy: {details.get('train_accuracy', 0):.3f} | CV: {details.get('cv_mean', 0):.3f}")

    return ht_model, df


def demo_do_operator_verification(ht_model):
    """Demonstrate DO operator methodology verification"""
    print("\n" + "="*80)
    print("DEMO 2: DO OPERATOR METHODOLOGY VERIFICATION")
    print("="*80)

    print("\n🔬 Verifying DO operator compliance...")

    # Create DO operator
    do_operator = DOOperator(
        graph=ht_model.graph,
        regressors_dict=ht_model.regressors_dict,
        baseline_stats=ht_model.baseline_stats,
        node_types=ht_model.node_types,
        label_encoders=getattr(ht_model, 'label_encoders', {})
    )

    # Test intervention: Increase marketing spend by 20%
    baseline_marketing = ht_model.baseline_stats['marketing_spend']['mean']
    intervention_value = baseline_marketing * 1.2

    print(f"\n   Test Intervention: marketing_spend")
    print(f"   Baseline: {baseline_marketing:.2f}")
    print(f"   Intervention: {intervention_value:.2f} (+20%)")

    # Apply DO operator
    result = do_operator.do(
        intervention_values={'marketing_spend': intervention_value}
    )

    print(f"\n   ✅ Intervention applied successfully!")
    print(f"\n   Affected nodes: {len(result.affected_nodes)}")
    print(f"      {sorted(result.affected_nodes)}")
    print(f"\n   Unaffected nodes: {len(result.unaffected_nodes)}")
    print(f"      {sorted(result.unaffected_nodes)}")

    # Show effects on key outcomes
    print(f"\n   📊 Causal Effects:")
    for node in ['customer_visits', 'sales']:
        if node in result.pct_changes:
            print(f"      {node}: {result.pct_changes[node]:+.2f}%")

    # Verify DO operator properties
    print(f"\n   🔍 Verifying DO operator properties...")
    verification = verify_do_operator_properties(
        do_operator,
        intervention_values={'marketing_spend': intervention_value}
    )

    print(f"\n   All checks passed: {verification['all_checks_passed']}")

    if not verification['all_checks_passed']:
        print("\n   ⚠️  Failed checks:")
        for check in verification['checks']:
            if not check['passed']:
                print(f"      - {check['check']}")
    else:
        print("   ✅ DO operator implementation is correct!")

    # Show causal paths
    print(f"\n   🔗 Causal paths from marketing_spend to sales:")
    paths = do_operator.get_all_causal_paths('marketing_spend', 'sales')
    for i, path in enumerate(paths, 1):
        print(f"      Path {i}: {' → '.join(path)}")

    return do_operator


def demo_timeseries_visualization(ht_model, df):
    """Demonstrate time series intervention visualization"""
    print("\n" + "="*80)
    print("DEMO 3: TIME SERIES INTERVENTION VISUALIZATION")
    print("="*80)

    print("\n📅 Preparing time series data...")

    # Generate fresh data with dates
    ts_df = generate_sample_data_with_categorical(n_samples=365, include_time_series=True)
    print(f"   Generated {len(ts_df)} days of data")
    print(f"   Date range: {ts_df['date'].min()} to {ts_df['date'].max()}")

    # Create time series analyzer
    ts_analyzer = TimeSeriesInterventionAnalyzer(
        graph=ht_model.graph,
        regressors_dict=ht_model.regressors_dict,
        baseline_stats=ht_model.baseline_stats,
        node_types=ht_model.node_types,
        label_encoders=getattr(ht_model, 'label_encoders', {})
    )

    # Simulate: "What if we increased marketing spend by 30% starting 90 days ago?"
    intervention_start_days_ago = 90

    print(f"\n   Simulating intervention:")
    print(f"   - What if we increased marketing_spend by +30%")
    print(f"   - Starting {intervention_start_days_ago} days ago")
    print(f"   - Outcome: sales")

    result = ts_analyzer.simulate_historical_intervention(
        historical_data=ts_df,
        intervention_node='marketing_spend',
        intervention_pct_change=30.0,  # +30%
        outcome_node='sales',
        intervention_start_date=intervention_start_days_ago,
        date_column='date'
    )

    # Generate report
    print(f"\n" + "─"*80)
    report = create_intervention_report(result, include_summary_stats=True)
    print(report)

    # Try to plot (if matplotlib available)
    try:
        print("\n   📈 Generating visualization...")
        fig, axes = ts_analyzer.plot_intervention_comparison(
            result=result,
            title="Marketing Spend Intervention: Actual vs Counterfactual"
        )

        # Save plot
        output_path = '/home/user/Intervention-Search/examples/intervention_timeseries.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved plot to: {output_path}")

    except ImportError:
        print("\n   ⚠️  Matplotlib not available - skipping plot generation")
    except Exception as e:
        print(f"\n   ⚠️  Could not generate plot: {str(e)}")

    # Export data
    print("\n   💾 Exporting counterfactual data...")
    try:
        export_path = '/home/user/Intervention-Search/examples/counterfactual_data.csv'
        ts_analyzer.export_counterfactual_data(
            result=result,
            output_path=export_path,
            format='csv'
        )
        print(f"   ✅ Exported data to: {export_path}")
    except Exception as e:
        print(f"   ⚠️  Could not export data: {str(e)}")

    # Analyze optimal timing
    print("\n   🎯 Analyzing optimal intervention timing...")
    print("   (This may take a minute...)")

    try:
        # Sample every 30 days to speed up
        timing_df = ts_df.iloc[::30].copy()

        optimal_timing = ts_analyzer.analyze_optimal_intervention_timing(
            historical_data=timing_df,
            intervention_node='marketing_spend',
            intervention_pct_change=30.0,
            outcome_node='sales',
            date_column='date',
            metric='cumulative_effect'
        )

        print(f"\n   Top 5 intervention start dates by cumulative effect:")
        print(optimal_timing.head(5)[['intervention_start_date', 'cumulative_effect', 'avg_effect']].to_string(index=False))

    except Exception as e:
        print(f"   ⚠️  Could not complete timing analysis: {str(e)}")

    return result


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("🚀 INTERVENTION SEARCH v2.1 - NEW FEATURES DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases:")
    print("  1. Ensemble training (multiple models per node)")
    print("  2. DO operator methodology verification")
    print("  3. Time series intervention visualization")

    # Demo 1: Ensemble Training
    ht_model, df = demo_ensemble_training()

    # Demo 2: DO Operator Verification
    do_operator = demo_do_operator_verification(ht_model)

    # Demo 3: Time Series Visualization
    ts_result = demo_timeseries_visualization(ht_model, df)

    # Final summary
    print("\n" + "="*80)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ Ensemble training automatically selects best model per node")
    print("  ✓ Categorical variables are handled properly")
    print("  ✓ DO operator methodology ensures causal correctness")
    print("  ✓ Time series visualization enables 'what if' historical analysis")
    print("\nNext Steps:")
    print("  - Check examples/intervention_timeseries.png for visualization")
    print("  - Review examples/counterfactual_data.csv for detailed data")
    print("  - Adapt this code for your own causal DAG and data")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
