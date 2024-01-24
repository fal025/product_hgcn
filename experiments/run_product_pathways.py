import pandas as pd
import subprocess

# Load the .tsv file in Pandas
data = pd.read_csv('unique_best_spaces_4.tsv', sep='\t')

# Get the desired columns
graph_num = data['Graph num']
e_copies = data['E copies']
h_copies = data['H copies']
s_copies = data['S copies']
e_dim = data['E dim']
h_dim = data['H dim']
s_dim = data['S dim']


# Iterate over the rows and generate the command line strings
for i in range(len(data)):
    total_dim = int(float(e_dim[i]) * int(e_copies[i]) + float(h_dim[i]) * int(h_copies[i]) + float(s_dim[i]) * int(s_copies[i])) 
    cmd_string = f"python train.py --task lp --dataset emb --model HGCN --manifold H{h_copies[i]}E{e_copies[i]}S{s_copies[i]} --lr 0.001 --dim 16 --num-layers 3 --dropout 0.2 --act sigmoid --bias 0 --optimizer RiemannianAdam --data-path ../chtc_curv/pathbank/pathbank_models/{graph_num[i]}.pt --edge-path ../chtc_curv/data/{graph_num[i]}/{graph_num[i]}_named.edges --log-freq 1000 --c 2 --total_dim {total_dim}"
    output_file = f"runs/pathway_runs/{graph_num[i]}.txt"
    print(i)
    with open(output_file, 'w') as fileout:
        subprocess.run(cmd_string, shell=True, stdout=fileout, stderr=subprocess.STDOUT)
    gcn_cmd_string = f"python train.py --task lp --dataset emb --model GCN --manifold Euclidean --lr 0.001 --dim 16 --num-layers 3 --dropout 0.2 --act sigmoid --bias 0 --optimizer RiemannianAdam --data-path ../chtc_curv/pathbank/pathbank_models/{graph_num[i]}.pt --edge-path ../chtc_curv/data/{graph_num[i]}/{graph_num[i]}_named.edges --log-freq 1000 --c 1 --total_dim {total_dim}"
    gcn_file = f"runs/pathway_runs/{graph_num[i]}_euc_gcn.txt"
    with open(gcn_file, 'w') as fileout:
        subprocess.run(cmd_string, shell=True, stdout=fileout, stderr=subprocess.STDOUT)
