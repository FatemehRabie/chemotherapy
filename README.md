# Integrated PDE and Reinforcement Learning Framework for Personalized Cancer Treatment Optimization

This repository provides tools for processing chemotherapy-related data, training reinforcement learning (RL) models, and evaluating their performance in simulation environments.

## Quick Start

### Prerequisites

1. Install required Python dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib stable-baselines3
   ```
2. Download the GDSC2 dataset (`GDSC2_fitted_dose_response_27Oct23.xlsx`) from [CancerRxGene](https://www.cancerrxgene.org/) and place it in the root directory of the repository.

### Steps to Run

#### 1. Process Data

Prepare the dataset for analysis using the following script:
```python
import pandas as pd

# Load the Excel file
file_path = './GDSC2_fitted_dose_response_27Oct23.xlsx'
df = pd.read_excel(file_path)

# Process data
df_subset = df[['CELL_LINE_NAME', 'DRUG_NAME', 'AUC']]
pivot_table = df_subset.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='AUC', aggfunc='size')
valid_cell_lines = pivot_table[pivot_table.notna().all(axis=1)].index
filtered_df = df_subset[df_subset['CELL_LINE_NAME'].isin(valid_cell_lines)]
df_no_duplicates = filtered_df.groupby(['CELL_LINE_NAME', 'DRUG_NAME'], as_index=False).agg({'AUC': 'mean'})

# Save processed data
df_no_duplicates.to_excel('./Filtered_GDSC2_No_Duplicates_Averaged.xlsx', index=False)
```

The processed data will be saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.

#### 2. Train RL Models

Train PPO, TRPO, and A2C models using the provided evaluation script:
```python
from utils.evaluation import evaluate

evaluate(['PPO', 'TRPO', 'A2C'], total_steps=100_000, num_steps=32, number_of_envs=4, number_of_eval_episodes=10, seed=19)
```

#### 3. Plot Results

Generate training performance plots with the following script:
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
plt.figure(figsize=(12, 8))

for algo in ['PPO', 'TRPO', 'A2C']:
    log_data = np.load(f'./logs_{algo}/evaluations.npz')
    ep_rewards = log_data['results']
    ep_rew_mean = ep_rewards.mean(axis=1)
    ep_rew_std = ep_rewards.std(axis=1)
    plt.plot(log_data['timesteps'], ep_rew_mean, label=algo)
plt.legend()
plt.savefig('mean_rewards_with_std.png')
```

This will create a plot `mean_rewards_with_std.png` showing training performance across algorithms.

### Outputs

- **Processed Data**: Saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.
- **Trained RL Models**: Evaluated with performance metrics.
- **Visualization**: Training performance plots saved as `mean_rewards_with_std.png`.
