"""
Generate Realistic Supply Chain Performance Data

Causal Graph:
    supplier_reliability → raw_material_quality → production_efficiency → on_time_delivery
    warehouse_capacity → inventory_turnover → order_fulfillment_time → on_time_delivery
    transportation_mode → shipping_cost
    transportation_mode → delivery_speed → on_time_delivery
    demand_variability → safety_stock → inventory_turnover
    lead_time → safety_stock
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_supply_chain_data(n_orders=800, save_path='notebook_examples/data/supply_chain_data.csv'):
    """Generate realistic supply chain performance data"""

    # Independent variables (exogenous)
    supplier_reliability = np.random.uniform(70, 99, n_orders)  # % on-time
    warehouse_capacity = np.random.uniform(5000, 50000, n_orders)  # cubic feet
    transportation_mode = np.random.choice(['Air', 'Ground', 'Sea'], n_orders, p=[0.2, 0.6, 0.2])
    demand_variability = np.random.uniform(5, 40, n_orders)  # coefficient of variation %
    lead_time = np.random.uniform(1, 30, n_orders)  # days
    order_quantity = np.random.randint(100, 5000, n_orders)  # units

    # Causal relationships

    # raw_material_quality depends on: supplier_reliability
    raw_material_quality = (
        50 + supplier_reliability * 0.45 +
        np.random.normal(0, 3, n_orders)
    )
    raw_material_quality = np.clip(raw_material_quality, 60, 100)

    # production_efficiency depends on: raw_material_quality
    production_efficiency = (
        60 + raw_material_quality * 0.35 +
        np.random.normal(0, 4, n_orders)
    )
    production_efficiency = np.clip(production_efficiency, 65, 98)

    # safety_stock depends on: demand_variability, lead_time
    safety_stock = (
        100 + demand_variability * 15 + lead_time * 8 +
        np.random.normal(0, 50, n_orders)
    )
    safety_stock = np.maximum(safety_stock, 50)

    # inventory_turnover depends on: warehouse_capacity, safety_stock
    inventory_turnover = (
        warehouse_capacity * 0.0008 - safety_stock * 0.002 +
        np.random.normal(0, 0.5, n_orders)
    )
    inventory_turnover = np.clip(inventory_turnover, 2, 20)  # 2-20 turns per year

    # delivery_speed depends on: transportation_mode
    speed_map = {'Air': 1.5, 'Ground': 5.0, 'Sea': 20.0}  # days
    delivery_speed = (
        np.array([speed_map[mode] for mode in transportation_mode]) +
        np.random.normal(0, 0.5, n_orders)
    )
    delivery_speed = np.maximum(delivery_speed, 0.5)

    # shipping_cost depends on: transportation_mode, order_quantity
    cost_per_unit = {'Air': 15, 'Ground': 5, 'Sea': 2}
    shipping_cost = (
        order_quantity * np.array([cost_per_unit[mode] for mode in transportation_mode]) +
        np.random.normal(0, 200, n_orders)
    )
    shipping_cost = np.maximum(shipping_cost, 100)

    # order_fulfillment_time depends on: inventory_turnover, production_efficiency
    order_fulfillment_time = (
        10 - inventory_turnover * 0.3 + (100 - production_efficiency) * 0.15 +
        np.random.normal(0, 1, n_orders)
    )
    order_fulfillment_time = np.clip(order_fulfillment_time, 1, 15)  # days

    # on_time_delivery depends on: production_efficiency, order_fulfillment_time, delivery_speed
    on_time_delivery_score = (
        production_efficiency * 0.4 +
        (15 - order_fulfillment_time) * 2 +
        (25 - delivery_speed) * 1.5 +
        np.random.normal(0, 5, n_orders)
    )
    on_time_delivery_score = np.clip(on_time_delivery_score, 60, 100)

    # Binary on-time delivery (>= 90 is on-time)
    on_time_delivery = (on_time_delivery_score >= 90).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'order_id': [f'ORD_{i:05d}' for i in range(n_orders)],
        'supplier_reliability': supplier_reliability.round(1),
        'warehouse_capacity': warehouse_capacity.round(0),
        'transportation_mode': transportation_mode,
        'demand_variability': demand_variability.round(2),
        'lead_time': lead_time.round(1),
        'order_quantity': order_quantity,
        'raw_material_quality': raw_material_quality.round(1),
        'production_efficiency': production_efficiency.round(1),
        'safety_stock': safety_stock.round(0),
        'inventory_turnover': inventory_turnover.round(2),
        'delivery_speed': delivery_speed.round(2),
        'shipping_cost': shipping_cost.round(2),
        'order_fulfillment_time': order_fulfillment_time.round(2),
        'on_time_delivery_score': on_time_delivery_score.round(1),
        'on_time_delivery': on_time_delivery
    })

    # Save
    df.to_csv(save_path, index=False)
    print(f"✓ Generated {len(df)} supply chain orders")
    print(f"✓ Saved to: {save_path}")
    print(f"\nData summary:")
    print(df.describe().round(2))

    return df

if __name__ == '__main__':
    df = generate_supply_chain_data()
