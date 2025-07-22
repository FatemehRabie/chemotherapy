import pandas as pd
from utils.evaluation import evaluate

def process_and_evaluate(file_path):
    # Load the Excel file
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

    for param in [0.0,0.001,0.01,0.1]:
        evaluate(['PPO','TRPO','A2C'], total_steps=40000, num_steps=32, beta=param, number_of_envs=4, number_of_eval_episodes=10, seed=19)

if __name__ == "__main__":
    file_path = './GDSC2_fitted_dose_response_27Oct23.xlsx'
    process_and_evaluate(file_path)