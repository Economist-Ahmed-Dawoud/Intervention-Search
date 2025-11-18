# Intervention Search - Notebook Examples

This folder contains comprehensive Jupyter notebook examples demonstrating the Intervention Search system across three different business domains.

## 📚 Available Notebooks

### 1. [Retail Store Optimization](1_retail_store_optimization.ipynb)
**Domain**: Retail
**Goal**: Increase store sales by 20%
**Key Features**:
- 500 retail stores dataset
- Multi-factor causal graph (location, marketing, staff, inventory)
- Single and multi-node intervention strategies
- Practical business recommendations

**Causal Structure**:
```
store_location → foot_traffic → sales
store_size → inventory_level → sales
marketing_spend → foot_traffic
price_discount → conversion_rate → sales
staff_count → customer_satisfaction → sales
competitor_proximity → foot_traffic
```

---

### 2. [Marketing Campaign Optimization](2_marketing_campaign_optimization.ipynb)
**Domain**: Digital Marketing
**Goal**: Maximize campaign conversions
**Key Features**:
- 600 marketing campaigns dataset
- End-to-end marketing funnel (impressions → clicks → conversions)
- Cost-effectiveness analysis
- Single vs multi-node strategy comparison

**Causal Structure**:
```
ad_budget → impressions → clicks → conversions
targeting_quality → click_through_rate → clicks
ad_creative_quality → click_through_rate
landing_page_quality → conversion_rate → conversions
audience_size → impressions
day_of_week → impressions
```

---

### 3. [Supply Chain Optimization](3_supply_chain_optimization.ipynb)
**Domain**: Supply Chain & Logistics
**Goal**: Improve on-time delivery by 15%
**Key Features**:
- 800 supply chain orders dataset
- Complex operational dependencies
- Implementation feasibility analysis
- Sensitivity analysis across different targets
- Phased action plan development

**Causal Structure**:
```
supplier_reliability → raw_material_quality → production_efficiency → on_time_delivery
warehouse_capacity → inventory_turnover → order_fulfillment_time → on_time_delivery
transportation_mode → delivery_speed → on_time_delivery
demand_variability → safety_stock → inventory_turnover
lead_time → safety_stock
```

---

## 🚀 Quick Start

### 1. Generate Data (First Time Only)

Run the data generation scripts to create realistic datasets:

```bash
cd notebook_examples

# Generate all datasets
python generate_retail_data.py
python generate_marketing_data.py
python generate_supply_chain_data.py
```

This will create CSV files in the `data/` folder.

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Open and Run

Open any of the three notebooks and run all cells sequentially.

---

## 📊 What Each Notebook Demonstrates

All notebooks showcase the **complete end-to-end workflow**:

1. **Data Loading**: Read pre-generated realistic data
2. **Causal Graph Definition**: Define domain-specific causal relationships
3. **Model Training**: Train HT causal models with quality assessment
4. **Intervention Search**: Find optimal interventions using:
   - Monte Carlo uncertainty propagation (1000 simulations)
   - Model quality gating
   - Bayesian optimization
   - Multi-objective ranking
5. **Analysis & Insights**:
   - Compare top interventions
   - Analyze causal paths
   - Evaluate feasibility
   - Generate business recommendations

---

## 🎯 Key Features Demonstrated

### Core Intervention Search Capabilities

✅ **Proper Uncertainty Quantification**
- Monte Carlo simulation (1000+ samples per intervention)
- Realistic confidence intervals
- Not naive RMSE × z-score

✅ **Model Quality Gating**
- R² based filtering of unreliable interventions
- Weakest-link principle for causal paths
- Automatic quality grading (A-F)

✅ **Bayesian Optimization**
- Adaptive search for optimal intervention values
- 3-5x faster than grid search
- Finds true optimal points

✅ **Causal Path Analysis**
- Path-level sensitivity analysis
- Identifies reliable vs unreliable paths
- Effect decomposition

✅ **Out-of-Distribution Detection**
- Flags interventions outside training data
- Prevents overconfident predictions
- Adjusts confidence scores

✅ **Multi-Objective Ranking**
- Balances accuracy, uncertainty, quality, simplicity
- Safety penalties for infeasible interventions
- Clear confidence scores

---

## 📁 Folder Structure

```
notebook_examples/
├── README.md                               # This file
├── 1_retail_store_optimization.ipynb       # Retail example
├── 2_marketing_campaign_optimization.ipynb # Marketing example
├── 3_supply_chain_optimization.ipynb       # Supply chain example
├── generate_retail_data.py                 # Retail data generator
├── generate_marketing_data.py              # Marketing data generator
├── generate_supply_chain_data.py           # Supply chain data generator
└── data/                                   # Generated datasets
    ├── retail_data.csv
    ├── marketing_data.csv
    └── supply_chain_data.csv
```

---

## 🎓 Learning Path

**Recommended order for beginners**:

1. **Start with Retail** (simplest causal structure)
2. **Move to Marketing** (introduces funnel dynamics)
3. **Finish with Supply Chain** (most complex, multi-stage operations)

Each notebook builds on concepts from the previous ones while introducing domain-specific considerations.

---

## 💡 Design Philosophy

These notebooks are designed to be:

- **Concise**: Focus on demonstrating the package, not lengthy data exploration
- **Clean**: Minimal styling code, maximum clarity
- **Realistic**: Data generation creates plausible business scenarios
- **Practical**: Clear business interpretation and actionable recommendations
- **Self-Contained**: Data is pre-generated, no external dependencies

---

## 🔧 Requirements

All notebooks require:

```python
# Core
numpy>=1.19.0
pandas>=1.1.0
networkx>=2.5.0
scikit-learn>=0.24.0
scipy>=1.5.0

# ML (for HT models)
xgboost>=1.3.0

# Notebook
jupyter>=1.0.0
```

Install from repository root:
```bash
pip install -r requirements.txt
```

---

## 📈 Expected Runtime

- **Data Generation**: ~5 seconds total for all three datasets
- **Notebook Execution**: ~2-3 minutes per notebook (with 1000 MC simulations)

---

## 🐛 Troubleshooting

### Issue: "No module named 'intervention_search'"

**Solution**: Run notebooks from the repository root or add to path:
```python
import sys
sys.path.insert(0, '..')
```

### Issue: "Data file not found"

**Solution**: Run data generation scripts first:
```bash
python generate_retail_data.py
python generate_marketing_data.py
python generate_supply_chain_data.py
```

### Issue: "Wide confidence intervals"

This is expected if:
- Model R² is low for some nodes
- Long causal chains (4+ hops)
- High inherent uncertainty

**Solution**: Check model quality metrics in the notebook output.

---

## 🤝 Contributing

To add new domain examples:

1. Create data generation script: `generate_<domain>_data.py`
2. Create notebook: `<number>_<domain>_optimization.ipynb`
3. Follow existing structure and style
4. Update this README

---

## 📚 Additional Resources

- **Main Documentation**: See [README.md](../README.md) in repository root
- **API Reference**: See `/intervention_search/api/` for detailed API docs
- **Example Scripts**: See `/examples/` for Python script examples

---

## 📝 Citation

If you use these examples in research or publications:

```bibtex
@software{intervention_search_examples,
  title={Intervention Search: Practical Examples for Causal Optimization},
  author={Causal AI Team},
  year={2024},
  note={Jupyter notebook examples for retail, marketing, and supply chain}
}
```

---

**Built to demonstrate robust, production-ready causal intervention search** 🎯
