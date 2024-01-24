import os
import re
import csv

# Define the directory where your log files are located
log_directory = 'runs/pathway_runs/'

# Create dictionaries to store the extracted values
results = {'graph_num': [], 'val_loss': [], 'val_roc': [], 'val_ap': [],
           'test_loss': [], 'test_roc': [], 'test_ap': []}

# Loop through the files in the directory
for filename in os.listdir(log_directory):
    if filename.endswith('_euc_gcn.txt'):
        with open(os.path.join(log_directory, filename), 'r') as file:
            lines = file.readlines()

            # Extract values from the last three lines of the file
            val_line = lines[-3]
            test_line = lines[-2]

            val_match = re.search(r'val_loss: (\d+\.\d+) val_roc: (\d+\.\d+) val_ap: (\d+\.\d+)', val_line)
            test_match = re.search(r'test_loss: (\d+\.\d+) test_roc: (\d+\.\d+) test_ap: (\d+\.\d+)', test_line)
            
            if val_match and test_match:
                val_loss, val_roc, val_ap = val_match.groups()
                test_loss, test_roc, test_ap = test_match.groups()

                # Extract the {num} from the filename
                num_match = re.search(r'(\d+)', filename)
                num = num_match.group(1) if num_match else 'Unknown'

                results['graph_num'].append(num)
                results['val_loss'].append(val_loss)
                results['val_roc'].append(val_roc)
                results['val_ap'].append(val_ap)
                results['test_loss'].append(test_loss)
                results['test_roc'].append(test_roc)
                results['test_ap'].append(test_ap)

# Define the names of the output CSV files
output_csv_filename = 'euclidean_gcn_results.csv'

# Write the extracted values to a CSV file
with open(output_csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['graph num', 'val_loss', 'val_roc', 'val_ap', 'test_loss', 'test_roc', 'test_ap'])
    for i in range(len(results['val_loss'])):
        writer.writerow([results['graph_num'][i], results['val_loss'][i], results['val_roc'][i], results['val_ap'][i],
                         results['test_loss'][i], results['test_roc'][i], results['test_ap'][i]])

print(f'Values extracted and saved to {output_csv_filename}')

# Create dictionaries to store the extracted values
results = {'graph_num': [], 'val_loss': [], 'val_roc': [], 'val_ap': [],
           'test_loss': [], 'test_roc': [], 'test_ap': []}

# Loop through the files in the directory
for filename in os.listdir(log_directory):
    if filename.endswith('.txt') and not filename.endswith('_euc_gcn.txt'):
        with open(os.path.join(log_directory, filename), 'r') as file:
            lines = file.readlines()

            # Extract values from the last three lines of the file
            val_line = lines[-3]
            test_line = lines[-2]

            val_match = re.search(r'val_loss: (\d+\.\d+) val_roc: (\d+\.\d+) val_ap: (\d+\.\d+)', val_line)
            test_match = re.search(r'test_loss: (\d+\.\d+) test_roc: (\d+\.\d+) test_ap: (\d+\.\d+)', test_line)
            
            if val_match and test_match:
                val_loss, val_roc, val_ap = val_match.groups()
                test_loss, test_roc, test_ap = test_match.groups()

                # Extract the {num} from the filename
                num_match = re.search(r'(\d+)', filename)
                num = num_match.group(1) if num_match else 'Unknown'

                results['graph_num'].append(num)
                results['val_loss'].append(val_loss)
                results['val_roc'].append(val_roc)
                results['val_ap'].append(val_ap)
                results['test_loss'].append(test_loss)
                results['test_roc'].append(test_roc)
                results['test_ap'].append(test_ap)

# Define the names of the output CSV files
output_csv_filename = 'product_gcn_results.csv'

# Write the extracted values to a CSV file
with open(output_csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['graph num', 'val_loss', 'val_roc', 'val_ap', 'test_loss', 'test_roc', 'test_ap'])
    for i in range(len(results['val_loss'])):
        writer.writerow([results['graph_num'][i], results['val_loss'][i], results['val_roc'][i], results['val_ap'][i],
                         results['test_loss'][i], results['test_roc'][i], results['test_ap'][i]])

print(f'Values extracted and saved to {output_csv_filename}')