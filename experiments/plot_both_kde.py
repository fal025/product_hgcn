import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# Load the data from the first CSV file
file1 = 'pathbank_prod_best_metrics.csv'  # Replace with the path to your first CSV file
df1 = pd.read_csv(file1, sep='\t')

# Load the data from the second CSV file
file2 = 'pathbank_n2v_best_metrics.csv'  # Replace with the path to your second CSV file
df2 = pd.read_csv(file2, sep='\t')

# Merge the two DataFrames on 'Graph #' and 'graph_num' to compare rows with matching graph numbers
# merged_df = pd.merge(df1, df2, left_on='Graph #', right_on='graph_num')
merged_df = pd.merge(df1, df2, left_on='Graph #', right_on='Graph #', suffixes=('_file1', '_file2'))

# Set up the seaborn style
sns.set(style="whitegrid")

# Create a single density plot with test ROC on the y-axis and test AP on the x-axis
plt.figure(figsize=(10, 10))
# sns.kdeplot(data=merged_df, x="Test ROC_file1", y="Test ROC_file2", color='blue', cmap='Blues', fill=True)
# sns.kdeplot(data=merged_df, x="Test AP_file1", y="Test AP_file2", color='red', cmap='Reds', fill=True)

# sns.kdeplot(data=merged_df, x="Val AP_file1", y="Val AP_file2", color='red', cmap='Reds', fill=True)
sns.kdeplot(data=merged_df, x="Val ROC_file1", y="Val ROC_file2", color='blue', cmap='Blues', fill=True)
# handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label="Product GCN"),
#            mpatches.Patch(facecolor=plt.cm.Blues(100), label="Euclidean GCN")]
# plt.legend(handles=handles)

# Add a green line indicating y = x
plt.plot([0, 1], [0, 1], c='green', linestyle='--', label='y=x')

# Set labels and title
# plt.xlabel('Test AP')
# plt.ylabel('Test ROC')
plt.xlabel('Product GCN')
plt.ylabel('Euclidean GCN (node2vec embeddings)')
plt.title('Validation ROC')
plt.savefig('imgs/pathbank/pathbank_comparison_scatter_sweep_n2v_val_roc.png')

# plt.grid(True)
plt.legend()
plt.show()
