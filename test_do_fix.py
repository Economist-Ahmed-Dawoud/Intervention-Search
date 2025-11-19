"""Quick test to verify DO operator fix"""
import sys
sys.path.insert(0, '/home/user/Intervention-Search')

import numpy as np
import networkx as nx
import pandas as pd

print("Testing DO operator fix...")

# Create simple test case
nodes = ['A', 'B', 'C', 'D']
edges = [('A', 'C'), ('B', 'D')]  # A->C, B->D (two independent chains)

# Create graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Mock regressors (simple identity function)
class MockRegressor:
    def predict(self, X):
        return X.mean(axis=1)

regressors_dict = {
    'C': (MockRegressor(), None),
    'D': (MockRegressor(), None)
}

baseline_stats = {
    'A': {'mean': 10.0},
    'B': {'mean': 20.0},
    'C': {'mean': 10.0},
    'D': {'mean': 20.0}
}

node_types = {n: 'continuous' for n in nodes}

# Import and test
from intervention_search.core.do_operator import DOOperator, verify_do_operator_properties

do_op = DOOperator(
    graph=G,
    regressors_dict=regressors_dict,
    baseline_stats=baseline_stats,
    node_types=node_types
)

# Intervene on A -> should only affect C, not B or D
result = do_op.do(intervention_values={'A': 15.0})

print(f"\nIntervention: A = 15.0 (from baseline 10.0)")
print(f"Expected: C should change, B and D should stay at baseline")
print(f"\nResults:")
print(f"  A: {result.post_intervention_values['A']:.2f} (intervened, should be 15.0)")
print(f"  B: {result.post_intervention_values['B']:.2f} (should be 20.0 - UNCHANGED)")
print(f"  C: {result.post_intervention_values['C']:.2f} (descendant, should change)")
print(f"  D: {result.post_intervention_values['D']:.2f} (should be 20.0 - UNCHANGED)")

# Verify
verification = verify_do_operator_properties(do_op, {'A': 15.0})

print(f"\n{'='*60}")
print(f"All Checks Passed: {verification['all_checks_passed']}")
print(f"{'='*60}")

for check in verification['checks']:
    status = "✅" if check['passed'] else "❌"
    print(f"{status} {check['check']}")
    if not check['passed']:
        print(f"    Expected: {check['expected']}, Actual: {check['actual']}")

if verification['all_checks_passed']:
    print(f"\n✅ DO operator fix successful!")
else:
    print(f"\n❌ DO operator still has issues")
    sys.exit(1)
