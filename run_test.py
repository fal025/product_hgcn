import os
import subprocess

dimension = [4, 8, 16, 32, 64]
dataset = ["cora", "pubmed", "airport"]
params = ["Euclidean", "PoincareBall", "Spherical", "S1S1", "P1P1", "S1P1", \
        "S1S1P1P1", "S2P1P1", "S1S1P2"]
tasks = ["nc", "lp"]

def run_test():
    for task in tasks:
        
        for dim in dimension:

            for data in dataset:

                for param in params:

                    command = "python3 train.py --task %s --dataset %s --model HypGCN --lr 0.01 --dim %d --num-layers 2 --act relu --bias 0 --dropout 0.5 --weight-decay 0.001 --manifold %s --log-freq 5 --cuda -1 --c 1" % (task, data, dim, param)
                    print(command)
                    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
                    stdout, stderr = process.communicate()
                    process.wait()

                    prefix = task + "_" + data + "_" + str(dim)+ "_" + "param"
                    stdout_name = prefix + ".out" 
                    stderr_name = prefix + ".err"
                    with open(stdout_name, "w") as out, open(stderr_name, "w") as err:

                        out.write(stdout)
                        err.write(stderr)

def run_dummy_test():

    task = "nc"
    data = "cora"
    dim = 16
    param = "S1S1P2"
    command = "python3 train.py --task %s --dataset %s --model HypGCN --lr 0.01 --dim %d --num-layers 2 --act relu --bias 0 --dropout 0.5 --weight-decay 0.001 --manifold %s --log-freq 5 --cuda -1 --c 1" % (task, data, dim, param)
    print(command)
    process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
    stdout, stderr = process.communicate()
    process.wait()

    prefix = task + "_" + data + "_" + str(dim)+ "_" + "param"
    stdout_name = prefix + ".out" 
    stderr_name = prefix + ".err"
    with open(stdout_name, "w") as out, open(stderr_name, "w") as err:

        out.write(str(stdout))
        err.write(str(stderr))


if __name__ == "__main__":
    
    #run_dummy_test()
    run_test()



                


            



