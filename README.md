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

# Extract only the relevant columns
df_subset = df[['CELL_LINE_NAME', 'DRUG_NAME', 'AUC']]

# Create a pivot table to check if each CELL_LINE_NAME has the same set of DRUG_NAME values
pivot_table = df_subset.pivot_table(index='CELL_LINE_NAME', columns='DRUG_NAME', values='AUC', aggfunc='size')

# Filter out rows with NaN values, ensuring all CELL_LINE_NAME have the same set of DRUG_NAME values
complete_rows_mask = pivot_table.notna().all(axis=1)
valid_cell_lines = pivot_table[complete_rows_mask].index

# Filter the original subset dataframe based on valid_cell_lines
filtered_df = df_subset[df_subset['CELL_LINE_NAME'].isin(valid_cell_lines)]

# Group by the first two columns and average the third column if duplicates are found
df_no_duplicates = filtered_df.groupby(['CELL_LINE_NAME', 'DRUG_NAME'], as_index=False).agg({'AUC': 'mean'})

# Save the updated DataFrame without duplicates to a new Excel file
output_file_path_no_duplicates = './Filtered_GDSC2_No_Duplicates_Averaged.xlsx'
df_no_duplicates.to_excel(output_file_path_no_duplicates, index=False)
```

The processed data will be saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.

#### 2. Train RL Models

Train PPO, TRPO, and A2C models using the provided evaluation script:
```python
from utils.evaluation import evaluate

for param in [0.01, 0.1, 0.5]:
    evaluate(['PPO','TRPO','A2C'], total_steps=40000, num_steps=32, beta=param, number_of_envs=4, number_of_eval_episodes=10, seed=19)
```

### Outputs

- **Processed Data**: Saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.
- **Trained RL Models**: Evaluated with performance metrics, for example the files in the folder `logs_PPO_0.01`.
- **Visualization**: Training performance plots, for example `rewards_beta_0.01.png`.
- **Sample test runs**: Saved in the folder `results`.
- **Summary of training metrics**: Such as wall-clock times, final and best-checkpoint performance, for example `A2C_0.01_training.txt`. 
