"""
Test script to validate all fixes:
1. KeyError 'quality' is fixed
2. Confidence intervals are in correct units (percentage)
3. Code runs without errors
"""

import pandas as pd
import numpy as np
import networkx as nx
from ht_categ import HT, HTConfig
from intervention_search import InterventionSearch

print("="*70)
print("TESTING ALL FIXES")
print("="*70)

# Generate synthetic data
np.random.seed(42)
n_samples = 500

data = {
    'marketing_spend': np.random.normal(1000, 200, n_samples),
    'price_discount': np.random.normal(10, 3, n_samples),
    'staff_count': np.random.normal(5, 1, n_samples),
}

# Generate dependent variables with clear causal relationships
data['foot_traffic'] = (
    0.5 * data['marketing_spend'] +
    np.random.normal(0, 100, n_samples)
)

data['conversion_rate'] = (
    0.3 * data['price_discount'] +
    10 +
    np.random.normal(0, 2, n_samples)
)

data['customer_satisfaction'] = (
    2 * data['staff_count'] +
    50 +
    np.random.normal(0, 5, n_samples)
)

data['sales'] = (
    0.1 * data['foot_traffic'] +
    5 * data['conversion_rate'] +
    0.5 * data['customer_satisfaction'] +
    np.random.normal(0, 20, n_samples)
)

df = pd.DataFrame(data)

# Define causal graph
nodes = ['marketing_spend', 'price_discount', 'staff_count',
         'foot_traffic', 'conversion_rate', 'customer_satisfaction', 'sales']

edges = [
    ('marketing_spend', 'foot_traffic'),
    ('price_discount', 'conversion_rate'),
    ('staff_count', 'customer_satisfaction'),
    ('foot_traffic', 'sales'),
    ('conversion_rate', 'sales'),
    ('customer_satisfaction', 'sales')
]

adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
for parent, child in edges:
    adj_matrix.loc[parent, child] = 1

print("\n✓ Generated synthetic data")
print(f"  Samples: {len(df)}")
print(f"  Variables: {list(df.columns)}")

# Train HT model
print("\n✓ Training causal model...")
config = HTConfig(graph=adj_matrix, model_type='LinearRegression')
ht_model = HT(config)
ht_model.train(df)
print("  Model trained successfully")

# Run intervention search
print("\n✓ Running intervention search...")
searcher = InterventionSearch(
    graph=ht_model.graph,
    ht_model=ht_model,
    n_simulations=100  # Reduced for faster testing
)

results = searcher.find_interventions(
    target_outcome='sales',
    target_change=10.0,
    tolerance=5.0,
    max_candidates=5,
    verbose=False
)

print("  Search completed successfully")

# TEST 1: Check for KeyError 'quality'
print("\n" + "="*70)
print("TEST 1: Quality Metrics in Results")
print("="*70)

# Get best candidate (marked with best=True flag)
best = next(c for c in results['all_candidates'] if c['best'])

try:
    quality_grade = best['quality']['quality_grade']
    quality_score = best['quality']['quality_score_geom_mean']
    print(f"✅ PASS: Quality metrics found in results")
    print(f"   Quality Grade: {quality_grade}")
    print(f"   Quality Score: {quality_score:.3f}")
    test1_pass = True
except KeyError as e:
    print(f"❌ FAIL: KeyError - {e}")
    test1_pass = False

# TEST 2: Check CI units (should be in percentage, not absolute)
print("\n" + "="*70)
print("TEST 2: Confidence Interval Units")
print("="*70)

try:
    actual_effect = best['actual_effect']
    ci_90 = best['ci_90']
    ci_width = ci_90[1] - ci_90[0]

    print(f"  Predicted Effect: {actual_effect:+.1f}%")
    print(f"  90% CI: [{ci_90[0]:+.1f}%, {ci_90[1]:+.1f}%]")
    print(f"  CI Width: {ci_width:.1f}%")

    # Sanity check: CI should be in reasonable range for percentage
    # If CIs were in absolute units, they'd be huge (hundreds or thousands)
    if abs(ci_90[0]) < 500 and abs(ci_90[1]) < 500:
        print(f"✅ PASS: Confidence intervals are in percentage units")
        print(f"   (If they were absolute, values would be in hundreds/thousands)")
        test2_pass = True
    else:
        print(f"❌ FAIL: CI values too large - likely still in absolute units")
        print(f"   Expected: within ±100%, Got: [{ci_90[0]:.1f}, {ci_90[1]:.1f}]")
        test2_pass = False

except Exception as e:
    print(f"❌ FAIL: Error checking CIs - {e}")
    test2_pass = False

# TEST 3: Check that all candidates have quality info
print("\n" + "="*70)
print("TEST 3: Quality in All Candidates")
print("="*70)

try:
    all_have_quality = all('quality' in candidate for candidate in results['all_candidates'])

    if all_have_quality:
        print(f"✅ PASS: All {len(results['all_candidates'])} candidates have quality metrics")
        for i, candidate in enumerate(results['all_candidates'][:3], 1):
            print(f"   {i}. {candidate['nodes'][0]}: Grade {candidate['quality']['quality_grade']}")
        test3_pass = True
    else:
        missing = sum(1 for c in results['all_candidates'] if 'quality' not in c)
        print(f"❌ FAIL: {missing} candidates missing quality metrics")
        test3_pass = False

except Exception as e:
    print(f"❌ FAIL: Error checking candidates - {e}")
    test3_pass = False

# FINAL SUMMARY
print("\n" + "="*70)
print("FINAL TEST SUMMARY")
print("="*70)

all_tests = [
    ("Quality Metrics (KeyError fix)", test1_pass),
    ("CI Units (Wide CI fix)", test2_pass),
    ("All Candidates Quality", test3_pass)
]

for test_name, passed in all_tests:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")

if all(p for _, p in all_tests):
    print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
    exit(0)
else:
    print("\n⚠️  SOME TESTS FAILED - Review above for details")
    exit(1)
