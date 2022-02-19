# %%
import yaml
import argparse
import numpy as np
import os
import pickle

from models import *
from experiment import VAEModule
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="config_file",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('-v',
                    type=int,
                    dest='version_num',
                    help='model version number',
                    default='0')

parser.add_argument('-ckp',
                    dest='ckp')

args = parser.parse_args()
with open(args.config_file, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

tt_logger = TestTubeLogger(
    save_dir=f"{config['logging_params']['save_dir']}/{config['exp_params']['dataset']}/",
    name=config['logging_params']['name'],
    debug=True,
    create_git_tag=False,
    version = args.version_num,
)

model = vae_models[config['model_params']['name']](**config['model_params'])


# load state dict from check point
if args.ckp:
    checkpoint = torch.load( args.ckp )

else:
    logger_path = f"{tt_logger.save_dir}/{tt_logger.name}/version_{args.version_num}"
    ckp_dir = f"{logger_path}/checkpoints/"
    ckp_file_num = len(os.listdir(ckp_dir))
    assert os.path.exists(ckp_dir), 'the checkpoint folder does not exist'
    assert ckp_file_num>0, 'the checkpoint file does not exist'
    ckp_name = [f for f in os.listdir(ckp_dir)][ckp_file_num-1] # use the lastest checkpoint if there is more than one

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        checkpoint = torch.load( os.path.join(ckp_dir, ckp_name) )
    else:
        device = torch.device("cpu")
        checkpoint = torch.load( os.path.join(ckp_dir, ckp_name), map_location=device )


new_state_dict = {}
for k in checkpoint['state_dict']:
    new_k = k.replace('model.', '')
    new_state_dict[new_k] = checkpoint['state_dict'][k]
model.load_state_dict(new_state_dict)


net = VAEModule(model, config['exp_params'])
net.freeze()



runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])


runner.test(net)


