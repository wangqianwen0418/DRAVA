# %%
import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

version = 0

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


# checkpoint = torch.load('logs/BetaVAE_B/version_6/checkpoints/_ckpt_epoch_3.ckpt')
# checkpoint = torch.load('logs/BetaVAE_B/version_0/checkpoints/_ckpt_epoch_9.ckpt')
checkpoint = torch.load( f"logs/{config['logging_params']['name']}/version_{version}/checkpoints/_ckpt_epoch_28.ckpt" )
new_state_dict = {}
for k in checkpoint['state_dict']:
    new_k = k.replace('model.', '')
    new_state_dict[new_k] = checkpoint['state_dict'][k]
model.load_state_dict(new_state_dict)

# model.simu_sample(10, torch.cuda.current_device())

net = VAEXperiment(model, config['exp_params'])
net.freeze()

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=True,
    create_git_tag=False,
    version = version,
)


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


