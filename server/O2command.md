
## GPU resources at O2
check gpu resources: `sinfo  --Format=nodehost,available,memory,statelong,gres:20 -p gpu`  
load cuda: `module load gcc/6.2.0 cuda/10.2`  

check docs about [enabling X11 forwarding](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1588662332/Using+X11+Applications+Remotely)
request interactive session with port forwarding: `srun --pty -t 2:0:0 --mem=50G -p interactive --x11 --tunnel 3000:3000 bash`  
or  
request GPU resources: `srun -n 1 --pty -c 4 -t 14:00:00 -p gpu --gres=gpu:1 --mem=50G bash`   

## create virtual pythone environment for the first time
`conda create --name vae_env python=3.7`  
`conda activate vae_env`
`pip install -r requirements.txt`

## run conda on O2
load conda: `module load conda2/4.2.13`  
activate conda: 
`conda deactivate && conda activate vae_env`

##
**train**

the sunspots datasets
`python run.py -c configs/bbvae.yaml`
the celeba datasets
`python run.py -c configs/bbvae_celeba.yaml`

**load_model**

the sunspots datasets
`python load_model.py -c configs/bbvae.yaml -v 0`
the celeba datasets
`python load_model.py -c configs/bbvae_celeba.yaml -v 0`

## old doc of pytorch lighting 
https://pytorch-lightning.readthedocs.io/en/0.6.0/

## host on aws ec2
ec2-user@ec2-3-16-49-23.us-east-2.compute.amazonaws.com
http://3.16.49.23:8080/