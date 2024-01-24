import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the first CSV file
file1 = 'pathbank_prod_best_metrics.tsv'  # Replace with the path to your first CSV file
df1 = pd.read_csv(file1, sep='\t')

# Load the data from the second CSV file
file2 = 'euclidean_metrics.csv'  # Replace with the path to your second CSV file
df2 = pd.read_csv(file2)

# Merge the two DataFrames on 'Graph #' to compare rows with matching graph numbers
merged_df = pd.merge(df1, df2, left_on='Graph #', right_on='graph_num')

# Plot test ROC vs. test AP for each file
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Test AP'], merged_df['Test ROC'], c='blue', label='Product GCN')
plt.scatter(merged_df['test_ap'], merged_df['test_roc'], c='red', label='Euclidean GCN')

# Add a green line indicating y = x
plt.plot([0, 1], [0, 1], c='green', linestyle='--', label='y=x')

# Set labels and legend
plt.xlabel('Test AP')
plt.ylabel('Test ROC')
plt.legend()

# Show the plot
plt.grid(True)
plt.title('Test ROC vs. Test AP')
plt.savefig('euclidean_vs_prod_comparison.png')
plt.show()
