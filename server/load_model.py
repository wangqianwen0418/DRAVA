# %%
import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEModule
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

VER_NUM = 1

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])

tt_logger = TestTubeLogger(
    save_dir=f"{config['logging_params']['save_dir']}/{config['exp_params']['dataset']}/",
    name=config['logging_params']['name'],
    debug=True,
    create_git_tag=False,
    version = VER_NUM,
)


# load state dict from check point
checkpoint_name = '_ckpt_epoch_29.ckpt'
ckp_path = f"{tt_logger.save_dir}/{tt_logger.name}/version_{VER_NUM}/checkpoints/{checkpoint_name}"
checkpoint = torch.load( ckp_path )
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


