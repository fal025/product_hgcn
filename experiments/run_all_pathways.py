import pandas as pd
import subprocess

# Load the .tsv file in Pandas
data = pd.read_csv('unique_best_spaces_4.tsv', sep='\t')

# Get the desired columns
graph_num = data['Graph num']
e_copies = data['E copies']
h_copies = data['H copies']
s_copies = data['S copies']

# Iterate over the rows and generate the command line strings
for i in range(len(data)):
    cmd_string = f"python train.py --task lp --dataset emb --model HypGCN --manifold E{e_copies[i]}H{h_copies[i]}S{s_copies[i]} --lr 0.01 --weight-decay 0.0005 --dim 16 --num-layers 2 --dropout 0.2 --act relu --bias 0 --optimizer Adam --data-path ../chtc_curv/pathbank/models/{graph_num[i]}.pt --edge-path ../chtc_curv/data/{graph_num[i]}/{graph_num[i]}_named.edges --log-freq 1000 --c 1"
    output_file = f"pathway_runs/{graph_num[i]}.txt"
    print(i)
    with open(output_file, 'w') as file:
        subprocess.run(cmd_string, shell=True, stdout=file, stderr=subprocess.STDOUT)
