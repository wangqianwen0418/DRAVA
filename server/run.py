import yaml
import argparse
import numpy as np
import subprocess
import pickle

from models import *
from experiment import VAEModule
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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


tt_logger = TestTubeLogger(
    save_dir=f"{config['logging_params']['save_dir']}/{config['exp_params']['dataset']}/",
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

# save config file for this experiment
# with open('config.yml', 'w') as outfile:
#             yaml.dump(config, outfile, default_flow_style=False)

model = vae_models[config['model_params']['name']](**config['model_params'])
myModule = VAEModule(model,config['exp_params'])

runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                    min_nb_epochs=1,
                    logger=tt_logger,
                    log_save_interval=200,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    check_val_every_n_epoch=10,
                    early_stop_callback = False,
                    **config['trainer_params'])
                    


# copy config file to the logger folder
logger_path = f"{tt_logger.save_dir}{tt_logger.name}/version_{tt_logger.experiment.version}/"
cmd = ["cp", args.filename, f"{logger_path}/config.yaml"]
subprocess.run(cmd)

with open(f"{logger_path}/model.pickle", "wb") as file_:
    pickle.dump(model, file_, -1)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(myModule)