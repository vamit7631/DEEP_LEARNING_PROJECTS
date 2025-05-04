import pandas as pd
import numpy as np
import random

# Define categories

sectors = ['Finance', 'Technology', 'Retail', 'Manufacturing', 'Healthcare']

n_samples = 1000

# Generate data
data = {
    'revenue': np.random.randint(100_000, 10_000_000, n_samples),
    'employee_count': np.random.randint(10, 1000, n_samples),
    'employee_age': np.random.randint(25, 60, n_samples),
    'spending_score' : np.round(np.random.uniform(0, 100, n_samples), 2),
    'sector_label': [random.choice(sectors) for _ in range(n_samples)]
}

# Create DataFrame
df = pd.DataFrame(data)

df['sector_label'] = df['sector_label'].apply(lambda x: 1 if x == 'Finance' else 0)

# Save to CSV (optional)
df.to_csv('./DATASETS/business_sector_dataset.csv', index=False)

# Show first few rows
print(df.head())
