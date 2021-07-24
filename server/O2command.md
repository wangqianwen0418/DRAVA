
### O2 commands
check gpu resources: `sinfo  --Format=nodehost,available,memory,statelong,gres:20 -p gpu`
load cuda: `module load gcc/6.2.0 cuda/10.2`
request interactive session: `srun --pty -t 2:0:0 --mem=2G -p interactive bash`
or request GPU resources: `srun -n 1 --pty -c 2 -t 4:00:00 -p gpu --gres=gpu:1 --mem=5G bash` 

load conda: `module load conda2/4.2.13`
activate conda: `conda activate vae_env`


### old doc of pytorch lighting 
https://pytorch-lightning.readthedocs.io/en/0.6.0/