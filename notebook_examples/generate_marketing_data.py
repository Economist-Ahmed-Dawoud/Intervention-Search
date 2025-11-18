"""
Generate Realistic Digital Marketing Campaign Data

Causal Graph:
    ad_budget → impressions → clicks → conversions
    targeting_quality → click_through_rate → clicks
    ad_creative_quality → click_through_rate
    landing_page_quality → conversion_rate → conversions
    audience_size → impressions
    day_of_week → impressions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_marketing_data(n_campaigns=600, save_path='notebook_examples/data/marketing_data.csv'):
    """Generate realistic digital marketing campaign data"""

    # Independent variables (exogenous)
    ad_budget = np.random.uniform(500, 20000, n_campaigns)  # $ per campaign
    targeting_quality = np.random.uniform(0, 100, n_campaigns)  # score 0-100
    ad_creative_quality = np.random.uniform(0, 100, n_campaigns)  # score 0-100
    landing_page_quality = np.random.uniform(0, 100, n_campaigns)  # score 0-100
    audience_size = np.random.randint(10000, 1000000, n_campaigns)
    day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                     'Friday', 'Saturday', 'Sunday'], n_campaigns)

    # Causal relationships

    # impressions depends on: budget, audience_size, day_of_week
    day_multiplier = {
        'Monday': 0.95, 'Tuesday': 1.0, 'Wednesday': 1.05,
        'Thursday': 1.1, 'Friday': 1.15, 'Saturday': 0.85, 'Sunday': 0.8
    }
    impressions = (
        ad_budget * 15 +
        audience_size * 0.02 +
        np.array([day_multiplier[d] for d in day_of_week]) * 5000 +
        np.random.normal(0, 1000, n_campaigns)
    )
    impressions = np.maximum(impressions, 100)

    # click_through_rate depends on: targeting_quality, ad_creative_quality
    click_through_rate = (
        0.5 +  # base CTR 0.5%
        targeting_quality * 0.03 +
        ad_creative_quality * 0.025 +
        np.random.normal(0, 0.3, n_campaigns)
    )
    click_through_rate = np.clip(click_through_rate, 0.1, 10)  # 0.1% - 10%

    # clicks depends on: impressions, click_through_rate
    clicks = (
        impressions * (click_through_rate / 100) +
        np.random.normal(0, 50, n_campaigns)
    )
    clicks = np.maximum(clicks, 1)

    # conversion_rate depends on: landing_page_quality
    conversion_rate = (
        1.0 +  # base conversion 1%
        landing_page_quality * 0.08 +
        np.random.normal(0, 0.5, n_campaigns)
    )
    conversion_rate = np.clip(conversion_rate, 0.5, 15)  # 0.5% - 15%

    # conversions depends on: clicks, conversion_rate
    conversions = (
        clicks * (conversion_rate / 100) +
        np.random.normal(0, 5, n_campaigns)
    )
    conversions = np.maximum(conversions, 0)

    # Calculate derived metrics
    cost_per_click = np.where(clicks > 0, ad_budget / clicks, 0)
    cost_per_acquisition = np.where(conversions > 0, ad_budget / conversions, 0)

    # Create DataFrame
    df = pd.DataFrame({
        'campaign_id': [f'CAMP_{i:04d}' for i in range(n_campaigns)],
        'ad_budget': ad_budget.round(2),
        'targeting_quality': targeting_quality.round(1),
        'ad_creative_quality': ad_creative_quality.round(1),
        'landing_page_quality': landing_page_quality.round(1),
        'audience_size': audience_size,
        'day_of_week': day_of_week,
        'impressions': impressions.round(0),
        'click_through_rate': click_through_rate.round(3),
        'clicks': clicks.round(0),
        'conversion_rate': conversion_rate.round(3),
        'conversions': conversions.round(1),
        'cost_per_click': cost_per_click.round(2),
        'cost_per_acquisition': cost_per_acquisition.round(2)
    })

    # Save
    df.to_csv(save_path, index=False)
    print(f"✓ Generated {len(df)} marketing campaigns")
    print(f"✓ Saved to: {save_path}")
    print(f"\nData summary:")
    print(df.describe().round(2))

    return df

if __name__ == '__main__':
    df = generate_marketing_data()
