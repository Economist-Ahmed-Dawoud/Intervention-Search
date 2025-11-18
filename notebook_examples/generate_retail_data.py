"""
Generate Realistic Retail Store Performance Data

Causal Graph:
    store_location → foot_traffic → sales
    store_size → inventory_level → sales
    marketing_spend → foot_traffic
    price_discount → conversion_rate → sales
    staff_count → customer_satisfaction → sales
    competitor_proximity → foot_traffic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_retail_data(n_stores=500, save_path='notebook_examples/data/retail_data.csv'):
    """Generate realistic retail store performance data"""

    # Independent variables (exogenous)
    store_location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_stores, p=[0.3, 0.5, 0.2])
    store_size = np.random.uniform(1000, 10000, n_stores)  # sq ft
    marketing_spend = np.random.uniform(1000, 15000, n_stores)  # monthly $
    price_discount = np.random.uniform(0, 30, n_stores)  # %
    staff_count = np.random.randint(3, 25, n_stores)
    competitor_proximity = np.random.uniform(0.5, 10, n_stores)  # miles

    # Causal relationships with realistic noise

    # foot_traffic depends on: location, marketing, competitors
    location_effect = {'Urban': 500, 'Suburban': 300, 'Rural': 150}
    foot_traffic = (
        np.array([location_effect[loc] for loc in store_location]) +
        marketing_spend * 0.8 -
        competitor_proximity * 15 +
        np.random.normal(0, 50, n_stores)
    )
    foot_traffic = np.maximum(foot_traffic, 50)  # minimum traffic

    # inventory_level depends on: store_size
    inventory_level = (
        store_size * 0.5 +
        np.random.normal(0, 200, n_stores)
    )
    inventory_level = np.maximum(inventory_level, 100)

    # conversion_rate depends on: price_discount
    conversion_rate = (
        15 + price_discount * 0.6 +  # base 15%, increases with discount
        np.random.normal(0, 2, n_stores)
    )
    conversion_rate = np.clip(conversion_rate, 5, 50)  # 5-50%

    # customer_satisfaction depends on: staff_count
    customer_satisfaction = (
        60 + staff_count * 1.2 +
        np.random.normal(0, 5, n_stores)
    )
    customer_satisfaction = np.clip(customer_satisfaction, 50, 100)

    # sales depends on: foot_traffic, inventory, conversion_rate, satisfaction
    sales = (
        foot_traffic * (conversion_rate / 100) * 80 +  # traffic × conversion
        inventory_level * 0.05 +  # inventory availability boost
        customer_satisfaction * 20 +  # satisfaction boost
        np.random.normal(0, 2000, n_stores)
    )
    sales = np.maximum(sales, 1000)  # minimum sales

    # Create DataFrame
    df = pd.DataFrame({
        'store_id': [f'STORE_{i:03d}' for i in range(n_stores)],
        'store_location': store_location,
        'store_size': store_size.round(0),
        'marketing_spend': marketing_spend.round(2),
        'price_discount': price_discount.round(1),
        'staff_count': staff_count,
        'competitor_proximity': competitor_proximity.round(2),
        'foot_traffic': foot_traffic.round(0),
        'inventory_level': inventory_level.round(0),
        'conversion_rate': conversion_rate.round(2),
        'customer_satisfaction': customer_satisfaction.round(1),
        'sales': sales.round(2)
    })

    # Save
    df.to_csv(save_path, index=False)
    print(f"✓ Generated {len(df)} retail stores")
    print(f"✓ Saved to: {save_path}")
    print(f"\nData summary:")
    print(df.describe().round(2))

    return df

if __name__ == '__main__':
    df = generate_retail_data()
