"""
Warehouse/Logistics Intervention Search Example

Demonstrates the new InterventionSearch system using warehouse operations data.

This example shows how to:
1. Load data and train causal models
2. Use the InterventionSearch API
3. Interpret results with proper uncertainty quantification
4. Make production-ready recommendations
"""

import pandas as pd
import numpy as np
import networkx as nx
import sys
sys.path.append('..')

# Import HT model (legacy)
from ht_categ import HT, HTConfig

# Import new InterventionSearch system
from intervention_search import InterventionSearch

print("="*70)
print("WAREHOUSE OPERATIONS - INTERVENTION SEARCH EXAMPLE")
print("="*70)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

print("\n📁 Step 1: Loading warehouse operations data...")

# Load the BHI dataset
df = pd.read_csv('../udm_1763133759721.csv')

# Calculate BHI metric
def calculate_bhi(df):
    df['bhi'] = (
        df['uphDept'] * 0.40 +
        df['ophDept'] * 0.30 +
        df['uphBuilding'] * 0.15 +
        df['ophBuilding'] * 0.15 +
        df['staffedHeadcount'] * 0.05 +
        df['systemDowntimeHrs'] * 0.02 +
        df['effectiveHours'] * 0.02 +
        df['areaUnitsReceiving'] * 0.02 +
        df['areaUnitsPutaway'] * 0.02 +
        df['areaUnitsAsrs'] * 0.02 +
        df['areaUnitsPicking'] * 0.02 +
        df['areaUnitsShipping'] * 0.02 +
        df['areaUnitsSortation'] * 0.02 +
        df['areaUnitsPacking'] * 0.02 +
        df['realizedUnits'] * 0.02 +
        df['forecastUnits'] * 0.02 +
        df['storageUtilizationPct'] * 0.02
    )
    return df

df = calculate_bhi(df)

print(f"   ✓ Loaded {len(df)} records")
print(f"   ✓ BHI range: [{df['bhi'].min():.1f}, {df['bhi'].max():.1f}]")

# ==============================================================================
# STEP 2: DEFINE CAUSAL DAG
# ==============================================================================

print("\n🔗 Step 2: Defining causal DAG...")

edges = [
    # Planning layer
    ('forecastUnits', 'staffedHeadcount'),
    ('forecastUnits', 'shifts'),
    ('targetOphBuilding', 'staffedHeadcount'),
    ('staffedHeadcount', 'staffedHours'),
    ('shifts', 'staffedHours'),

    # Workload layer
    ('orders', 'realizedUnits'),
    ('singlesShare', 'realizedUnits'),
    ('nonconShare', 'effectiveHours'),
    ('bulkShare', 'effectiveHours'),
    ('unitsPerVendorCarton', 'effectiveHours'),
    ('systemDowntimeHrs', 'effectiveHours'),
    ('staffedHours', 'effectiveHours'),
    ('realizedUnits', 'effectiveHours'),
    ('orders', 'effectiveHours'),

    # Operations layer
    ('realizedUnits', 'areaUnitsReceiving'),
    ('realizedUnits', 'areaUnitsPutaway'),
    ('realizedUnits', 'areaUnitsPicking'),
    ('realizedUnits', 'areaUnitsSortation'),
    ('realizedUnits', 'areaUnitsAsrs'),
    ('realizedUnits', 'areaUnitsPacking'),
    ('realizedUnits', 'areaUnitsShipping'),

    # Storage layer
    ('areaUnitsReceiving', 'storagePallets'),
    ('areaUnitsShipping', 'storagePallets'),
    ('storagePallets', 'storageUtilizationPct'),

    # KPI layer
    ('realizedUnits', 'uphDept'),
    ('effectiveHours', 'uphDept'),
    ('orders', 'ophDept'),
    ('effectiveHours', 'ophDept'),

    # Building aggregation
    ('realizedUnits', 'buildingRealizedUnits'),
    ('orders', 'buildingOrders'),
    ('effectiveHours', 'buildingEffectiveHours'),

    # Building KPIs
    ('buildingRealizedUnits', 'uphBuilding'),
    ('buildingEffectiveHours', 'uphBuilding'),
    ('buildingOrders', 'ophBuilding'),
    ('buildingEffectiveHours', 'ophBuilding'),

    # Variance
    ('forecastUnits', 'keyAssumptionVariance'),
    ('realizedUnits', 'keyAssumptionVariance'),
    ('nonconShare', 'keyAssumptionVariance'),
    ('singlesShare', 'keyAssumptionVariance'),

    # BHI (outcome)
    ('uphDept', 'bhi'),
    ('ophDept', 'bhi'),
    ('uphBuilding', 'bhi'),
    ('ophBuilding', 'bhi'),
    ('staffedHeadcount', 'bhi'),
    ('systemDowntimeHrs', 'bhi'),
    ('effectiveHours', 'bhi'),
    ('areaUnitsReceiving', 'bhi'),
    ('areaUnitsPutaway', 'bhi'),
    ('areaUnitsAsrs', 'bhi'),
    ('areaUnitsPicking', 'bhi'),
    ('areaUnitsShipping', 'bhi'),
    ('areaUnitsSortation', 'bhi'),
    ('areaUnitsPacking', 'bhi'),
    ('realizedUnits', 'bhi'),
    ('forecastUnits', 'bhi'),
    ('storageUtilizationPct', 'bhi')
]

# Create graph
G = nx.DiGraph(edges)
unique_variables = sorted(list(G.nodes()))

print(f"   ✓ DAG has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"   ✓ Is valid DAG: {nx.is_directed_acyclic_graph(G)}")

# Create adjacency matrix
adj_df = pd.DataFrame(0, index=unique_variables, columns=unique_variables)
for source, target in edges:
    adj_df.loc[source, target] = 1

# ==============================================================================
# STEP 3: TRAIN CAUSAL MODEL
# ==============================================================================

print("\n🎓 Step 3: Training causal models...")

# Filter data
df_full = df[unique_variables].copy().dropna()

# Split into normal vs anomalous
anomaly_threshold = df_full['bhi'].quantile(0.10)
df_train = df_full[df_full['bhi'] >= anomaly_threshold].copy()
df_anomaly = df_full[df_full['bhi'] < anomaly_threshold].copy()

print(f"   ✓ Training data: {len(df_train)} samples")
print(f"   ✓ Anomaly data: {len(df_anomaly)} samples")

# Train HT model
config = HTConfig(
    graph=adj_df,
    aggregator="max",
    root_cause_top_k=5,
    model_type='Xgboost'
)

ht_model = HT(config)
ht_model.train(df_train, perform_cv=True)

print(f"   ✓ Models trained!")

# Get quality report
quality_report = ht_model.get_model_quality_report()
print(f"   ✓ Model quality grade: {quality_report['trust_indicators']['quality_grade']}")
print(f"   ✓ Mean R²: {quality_report['overall_summary']['regression_performance']['mean_r2']:.3f}")

# ==============================================================================
# STEP 4: INITIALIZE INTERVENTION SEARCH (NEW SYSTEM!)
# ==============================================================================

print("\n🚀 Step 4: Initializing InterventionSearch system...")

searcher = InterventionSearch(
    graph=ht_model.graph,
    ht_model=ht_model,
    n_simulations=1000,  # Monte Carlo simulations
    strict_quality_mode=False,
    random_seed=42
)

print("   ✓ InterventionSearch initialized with:")
print(f"      - Monte Carlo propagation (1000 simulations)")
print(f"      - Model quality gating")
print(f"      - Path sensitivity analysis")
print(f"      - Out-of-distribution detection")
print(f"      - Bayesian optimization")

# ==============================================================================
# STEP 5: FIND OPTIMAL INTERVENTIONS
# ==============================================================================

print("\n" + "="*70)
print("SEARCHING FOR OPTIMAL INTERVENTIONS")
print("="*70)

# Example 1: Improve BHI by 10%
print("\n🎯 Use Case 1: Improve BHI by 10%")
print("-" * 70)

results_bhi = searcher.find_interventions(
    target_outcome='bhi',
    target_change=+10.0,
    tolerance=2.0,
    max_intervention_pct=25.0,
    allow_combinations=True,
    max_candidates=5,
    confidence_level=0.90,
    verbose=True
)

# Example 2: Improve area units receiving by 15%
print("\n" + "="*70)
print("\n🎯 Use Case 2: Improve areaUnitsReceiving by 15%")
print("-" * 70)

results_receiving = searcher.find_interventions(
    target_outcome='areaUnitsReceiving',
    target_change=+15.0,
    tolerance=3.0,
    max_intervention_pct=30.0,
    allow_combinations=False,
    max_candidates=5,
    verbose=True
)

# ==============================================================================
# STEP 6: ANALYZE RESULTS
# ==============================================================================

print("\n" + "="*70)
print("DETAILED ANALYSIS OF BEST INTERVENTION")
print("="*70)

# Get best candidate (marked with best=True flag)
best = next((c for c in results_bhi['all_candidates'] if c['best']), None)

if best:
    print(f"\n📊 Intervention Details:")
    print(f"   Type: {best['intervention_type']}")
    print(f"   Variables: {', '.join(best['nodes'])}")
    print(f"\n📝 Required Changes:")
    for node, pct in best['required_pct_changes'].items():
        baseline_val = ht_model.baseline_stats[node]['mean']
        new_val = baseline_val * (1 + pct / 100)
        print(f"   • {node}:")
        print(f"      - Current (baseline): {baseline_val:.2f}")
        print(f"      - Required change: {pct:+.2f}%")
        print(f"      - New target value: {new_val:.2f}")

    print(f"\n📈 Expected Outcomes:")
    print(f"   • Point Estimate: {best['actual_effect']:+.2f}%")
    print(f"   • 90% Confidence Interval: [{best['ci_90'][0]:+.2f}%, {best['ci_90'][1]:+.2f}%]")
    print(f"   • 50% Confidence Interval: [{best['ci_50'][0]:+.2f}%, {best['ci_50'][1]:+.2f}%]")
    print(f"   • Uncertainty (std): ±{best['prediction_uncertainty_std']:.2f}%")

    print(f"\n🎯 Reliability Metrics:")
    print(f"   • Confidence Score: {best['confidence']:.1%}")
    print(f"   • Within Tolerance: {'Yes ✅' if best['within_tolerance'] else 'No ❌'}")
    print(f"   • Error from Target: {best['error_from_target']:.2f}%")
    print(f"   • Search Iterations: {best['search_iterations']}")

    # Validation info
    if 'validation' in best:
        val = best['validation']
        print(f"\n✓ Validation Results:")
        print(f"   • Feasible: {'Yes ✅' if val['is_feasible'] else 'No ❌'}")
        print(f"   • Safe (In-Distribution): {'Yes ✅' if val['is_safe'] else 'No ⚠️'}")
        if val['warnings']:
            print(f"   • Warnings:")
            for warning in val['warnings'][:3]:
                print(f"      - {warning}")

    # Path analysis
    path_analysis = results_bhi.get('path_analysis')
    if path_analysis:
        print(f"\n🛤️  Causal Path Analysis:")
        print(f"   • Total paths: {path_analysis['num_paths']}")
        if path_analysis['most_reliable_path']:
            mrp = path_analysis['most_reliable_path']
            print(f"   • Most reliable path: {mrp['path']}")
            print(f"   • Path quality: {mrp['quality']:.2f}")
            print(f"   • Path length: {mrp['length']} hops")

# ==============================================================================
# STEP 7: COMPARE MULTIPLE OPTIONS
# ==============================================================================

print("\n" + "="*70)
print("COMPARING TOP 5 INTERVENTION OPTIONS")
print("="*70)

all_candidates = results_bhi.get('all_candidates', [])

if all_candidates:
    print(f"\n{'Rank':<6} {'Type':<12} {'Variables':<30} {'Effect':<10} {'90% CI':<25} {'Confidence'}")
    print("-" * 105)

    for i, candidate in enumerate(all_candidates[:5], 1):
        vars_str = ', '.join(candidate['nodes'])
        if len(vars_str) > 28:
            vars_str = vars_str[:25] + '...'

        ci_str = f"[{candidate['ci_90'][0]:+.1f}%, {candidate['ci_90'][1]:+.1f}%]"

        print(f"{i:<6} {candidate['intervention_type']:<12} {vars_str:<30} "
              f"{candidate['actual_effect']:+.2f}%    {ci_str:<25} {candidate['confidence']:.0%}")

# ==============================================================================
# STEP 8: RECOMMENDATIONS
# ==============================================================================

print("\n" + "="*70)
print("💡 ACTIONABLE RECOMMENDATIONS")
print("="*70)

if best:
    status = best.get('within_tolerance', False)

    if status:
        print("\n✅ APPROVED FOR IMPLEMENTATION")
        print(f"\nRecommendation:")
        for node, pct in best['required_pct_changes'].items():
            baseline = ht_model.baseline_stats[node]['mean']
            target = baseline * (1 + pct / 100)
            print(f"  → Adjust '{node}' by {pct:+.2f}% (from {baseline:.2f} to {target:.2f})")

        print(f"\nExpected Outcome:")
        print(f"  → BHI will improve by {best['actual_effect']:+.2f}%")
        print(f"  → 90% confidence that improvement will be between")
        print(f"     {best['ci_90'][0]:+.2f}% and {best['ci_90'][1]:+.2f}%")

        print(f"\nConfidence Level: {best['confidence']:.0%}")

        if best['confidence'] >= 0.8:
            print("  ✓ High confidence - proceed with implementation")
        elif best['confidence'] >= 0.6:
            print("  ⚠️  Moderate confidence - consider pilot test first")
        else:
            print("  ⚠️  Lower confidence - recommend monitoring and validation")

    else:
        print("\n⚠️  CAUTION RECOMMENDED")
        print(f"\nThe best intervention found has an error of {best['error_from_target']:.2f}%")
        print("which exceeds the tolerance threshold.")
        print("\nConsider:")
        print("  1. Relaxing the tolerance constraints")
        print("  2. Collecting more training data to improve model quality")
        print("  3. Testing whether the target is realistic given system dynamics")

# ==============================================================================
# STEP 9: SUMMARY STATISTICS
# ==============================================================================

print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

summary = results_bhi['summary']

print(f"\nSearch Performance:")
print(f"   • Total interventions tested: {summary['total_tested']}")
print(f"   • Passed validation: {summary['passed_validation']}")
print(f"   • Within tolerance: {summary['within_tolerance']}")
print(f"   • High confidence (≥70%): {summary['high_confidence']}")
print(f"   • Target achieved: {'Yes ✅' if summary['target_achieved'] else 'No ❌'}")

quality_report = results_bhi['quality_report']
print(f"\nModel Quality:")
print(f"   • Overall grade: {quality_report['overall_grade']}")
print(f"   • Mean R²: {quality_report['mean_r2']:.3f}")
print(f"   • High-quality models: {quality_report['high_quality_count']}/{quality_report['total_models']}")

if quality_report['low_quality_count'] > 0:
    print(f"   ⚠️  Low-quality models detected:")
    for node in quality_report['low_quality_nodes'][:5]:
        print(f"      - {node}")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)

print("\n📝 Key Improvements Over Original System:")
print("   1. ✅ Proper uncertainty quantification (Monte Carlo)")
print("   2. ✅ Model quality gating (reject unreliable paths)")
print("   3. ✅ Out-of-distribution detection")
print("   4. ✅ Smarter search (Bayesian optimization)")
print("   5. ✅ Path sensitivity analysis")
print("   6. ✅ Multi-objective ranking")
print("\n   Result: Narrower confidence intervals, higher reliability!")
print("="*70)
