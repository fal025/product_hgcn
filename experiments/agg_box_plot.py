
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Initialize an empty list to store DataFrames for each dataset
dataset_dataframes = []

# List of dataset names
dataset_names = ["pathbank", "reactome", "kegg", "humancyc", "nci"]

# List of model names
model_names = ['Product GCN', 'Laplacian GCN', 'Euclidean GCN', 'Node2Vec GCN']

# Loop through each dataset
for dataset_name in dataset_names:
    # Initialize an empty list to store DataFrames for each CSV file in the dataset
    dataframes = []

    # List of CSV file paths for the current dataset
    csv_files = [
        f'{dataset_name}/{dataset_name}_prod_best_metrics.csv',
        f'{dataset_name}/{dataset_name}_laplacian_best_metrics.csv',
        f'{dataset_name}/{dataset_name}_euc_best_metrics.csv',
        f'{dataset_name}/{dataset_name}_n2v_best_metrics.csv'
    ]


    # Loop through each CSV file in the dataset
    for i, csv_file in enumerate(csv_files):
        # Check if the CSV file exists
        if os.path.exists(csv_file):
            # Read the CSV file into a DataFrame and append it to the list
            df = pd.read_csv(csv_file, sep='\t')
            df['Model'] = model_names[i]
            dataframes.append(df)

    # Combine the DataFrames from all CSV files in the dataset into one DataFrame
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        # Add a 'Dataset' column to distinguish datasets
        merged_df['Dataset'] = dataset_name
        dataset_dataframes.append(merged_df)

# Combine the DataFrames from all datasets into one DataFrame
all_datasets_merged_df = pd.concat(dataset_dataframes, ignore_index=True)

# Create a boxplot for the aggregated results across datasets
plt.figure(figsize=(12, 6))
sns.boxplot(data=all_datasets_merged_df, y="Val ROC", x="Model")
plt.title('Validation AUROC - All Pathway Databases')

# Save the plot
plt.savefig('imgs/all/val_roc_all_datasets.png')

# Show the plot
plt.show()

