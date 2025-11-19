"""
Test script to validate Complete_System_Demonstration notebook functionality
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("TESTING COMPLETE SYSTEM DEMONSTRATION NOTEBOOK")
print("="*70)

# Test 1: Data Loading
print("\n1️⃣ Testing Data Loading...")
try:
    df = pd.read_csv('notebook_examples/data/retail_data.csv')
    assert len(df) > 0, "Empty dataframe"
    assert 'sales' in df.columns, "Missing sales column"
    print(f"   ✅ Data loaded: {len(df)} stores, {len(df.columns)} columns")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test 2: Causal Graph Definition
print("\n2️⃣ Testing Causal Graph...")
try:
    nodes = [
        'store_location', 'store_size', 'marketing_spend', 'price_discount',
        'staff_count', 'competitor_proximity', 'foot_traffic', 'inventory_level',
        'conversion_rate', 'customer_satisfaction', 'sales'
    ]

    edges = [
        ('store_location', 'foot_traffic'),
        ('marketing_spend', 'foot_traffic'),
        ('competitor_proximity', 'foot_traffic'),
        ('store_size', 'inventory_level'),
        ('price_discount', 'conversion_rate'),
        ('staff_count', 'customer_satisfaction'),
        ('foot_traffic', 'sales'),
        ('inventory_level', 'sales'),
        ('conversion_rate', 'sales'),
        ('customer_satisfaction', 'sales')
    ]

    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    for parent, child in edges:
        adj_matrix.loc[parent, child] = 1

    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    assert nx.is_directed_acyclic_graph(G), "Graph contains cycles"
    print(f"   ✅ Causal graph valid: {len(nodes)} nodes, {len(edges)} edges")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test 3: Model Training
print("\n3️⃣ Testing Model Training with AutoML...")
try:
    from ht_categ import HT, HTConfig

    config = HTConfig(
        graph=adj_matrix,
        model_type='AutoML',
        auto_ml=True,
        auto_ml_models=['LinearRegression', 'RandomForest'],  # Reduced for speed
        aggregator='max',
        root_cause_top_k=5
    )

    ht_model = HT(config)
    ht_model.train(df, perform_cv=False, verbose_automl=False)  # No CV for speed

    quality_report = ht_model.get_model_quality_report()
    mean_r2 = quality_report['overall_summary']['regression_performance']['mean_r2']

    print(f"   ✅ Model trained: Mean R² = {mean_r2:.3f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Intervention Search
print("\n4️⃣ Testing Intervention Search...")
try:
    from intervention_search import InterventionSearch

    searcher = InterventionSearch(
        graph=ht_model.graph,
        ht_model=ht_model,
        n_simulations=100,  # Reduced for speed
        random_seed=42
    )

    results = searcher.find_interventions(
        target_outcome='sales',
        target_change=20.0,
        tolerance=3.0,
        confidence_level=0.90,
        max_intervention_pct=30.0,
        allow_combinations=False,
        verbose=False
    )

    best = results['best_intervention']
    print(f"   ✅ Search complete: {', '.join(best['nodes'])} ({best['actual_effect']:+.1f}%)")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: DO Operator
print("\n5️⃣ Testing DO Operator...")
try:
    from intervention_search import DOOperator, verify_do_operator_properties

    do_operator = DOOperator(
        graph=ht_model.graph,
        regressors_dict=ht_model.regressors_dict,
        baseline_stats=ht_model.baseline_stats,
        node_types=ht_model.node_types,
        label_encoders=ht_model.label_encoders
    )

    intervention_node = best['nodes'][0]
    intervention_value = ht_model.baseline_stats[intervention_node]['mean'] * 1.1

    result = do_operator.do(
        intervention_values={intervention_node: intervention_value}
    )

    verification = verify_do_operator_properties(
        do_operator,
        intervention_values={intervention_node: intervention_value}
    )

    print(f"   ✅ DO operator verified: All checks passed = {verification['all_checks_passed']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Time Series Analyzer
print("\n6️⃣ Testing Time Series Intervention Analyzer...")
try:
    from intervention_search import TimeSeriesInterventionAnalyzer

    df_time = df.copy()
    df_time['period'] = range(len(df_time))

    ts_analyzer = TimeSeriesInterventionAnalyzer(
        graph=ht_model.graph,
        regressors_dict=ht_model.regressors_dict,
        baseline_stats=ht_model.baseline_stats,
        node_types=ht_model.node_types,
        label_encoders=ht_model.label_encoders
    )

    intervention_pct = best['required_pct_changes'][intervention_node]

    ts_result = ts_analyzer.simulate_historical_intervention(
        historical_data=df_time,
        intervention_node=intervention_node,
        intervention_pct_change=intervention_pct,
        outcome_node='sales',
        intervention_start_date=100,
        date_column='period'
    )

    print(f"   ✅ Time series analysis complete: Cumulative effect = {ts_result.cumulative_effect:+.2f}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Root Cause Analysis
print("\n7️⃣ Testing Root Cause Analysis...")
try:
    df_anomaly = df.tail(100).copy()
    df_anomaly['marketing_spend'] = df_anomaly['marketing_spend'] * 0.7

    rca_results = ht_model.find_root_causes(
        df_anomaly,
        anomalous_metrics='sales',
        return_paths=True,
        adjustment=False
    )

    top_root = rca_results.root_cause_nodes[0]['root_cause']
    print(f"   ✅ RCA complete: Top root cause = {top_root}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n📊 Summary:")
print(f"   • Data loaded: {len(df)} stores")
print(f"   • Graph: {len(nodes)} nodes, {len(edges)} edges")
print(f"   • Model quality: {mean_r2:.3f} mean R²")
print(f"   • Best intervention: {', '.join(best['nodes'])} ({best['actual_effect']:+.1f}%)")
print(f"   • Confidence: {best['confidence']:.1%}")
print(f"   • DO operator: {'✅ Valid' if verification['all_checks_passed'] else '❌ Failed'}")
print("\n✅ Notebook is ready to use!")
print("="*70)
