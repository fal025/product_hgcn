import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file1 = 'pathbank_prod_best_metrics.csv'  # Replace with the path to your first CSV file
df1 = pd.read_csv(file1, sep='\t')

# Load the data from the second CSV file
file2 = 'pathbank_laplacian_best_metrics.csv'  # Replace with the path to your second CSV file
df2 = pd.read_csv(file2, sep='\t')

# Load the data from the second CSV file
file3 = 'pathbank_euc_best_metrics.csv'  # Replace with the path to your second CSV file
df3 = pd.read_csv(file3, sep='\t')

file4 = 'pathbank_n2v_best_metrics.csv'  # Replace with the path to your second CSV file
df4 = pd.read_csv(file4, sep='\t')

df1['Model'] = 'Product GCN'
df2['Model'] = 'Laplacian GCN'
df3['Model'] = 'Euclidean GCN'
df4['Model'] = 'Node2Vec GCN'
merged_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
sns.boxplot(data=merged_df, y="Test ROC", x="Model")
plt.title('Test ROC')
plt.savefig('imgs/pathbank/test_roc_all_pathbank.png')
plt.show()