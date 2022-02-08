import yaml
import argparse
import numpy as np
import shutil
import os


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

parser.add_argument('-n',
                    dest="n_epoch",
                    type=int,
                    help =  'validate every n epochs',
                    default=10)

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

model = vae_models[config['model_params']['name']](**config['model_params'])
myModule = VAEModule(model,config['exp_params'])

# checkpoint_callback = ModelCheckpoint(
#     filepath = f"{tt_logger.save_dir}{tt_logger.name}/version_{tt_logger.experiment.version}/checkpoints",
#     verbose=True,
#     save_top_k= -1, # save at each epoch
#     monitor='val_loss',
#     mode='min',
#     prefix=''
# )

runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                    min_nb_epochs=1,
                    logger=tt_logger,
                    log_save_interval=200,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=5,
                    check_val_every_n_epoch=args.n_epoch,
                    # checkpoint_callback=checkpoint_callback,
                    early_stop_callback = False,
                    **config['trainer_params'])

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = torch.device("cpu")                  


# copy config file to the logger folder
logger_path = f"{tt_logger.save_dir}{tt_logger.name}/version_{tt_logger.experiment.version}/"
shutil.copy(args.filename, os.path.join(logger_path, 'config.yaml'))

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(myModule)

print("=========test==================")
# test with the best model if checkpoin exist
# load state dict from check point
logger_path = f"{tt_logger.save_dir}/{tt_logger.name}/version_{tt_logger.experiment.version}"
ckp_dir = f"{logger_path}/checkpoints/"
ckp_file_num = len(os.listdir(ckp_dir))
if (not os.path.exists(ckp_dir) ) or ckp_file_num == 0:
    print('==========no checkpoint, use the latest model===========')
    runner.test(myModule) 

else:
    
    ckp_name = [f for f in os.listdir(ckp_dir)][ckp_file_num-1] # use the lastest checkpoint if there is more than one
    print(f'==========found checkpoint {ckp_name }===========')
    checkpoint = torch.load( os.path.join(ckp_dir, ckp_name) , map_location=device)
    new_state_dict = {}
    for k in checkpoint['state_dict']:
        new_k = k.replace('model.', '')
        new_state_dict[new_k] = checkpoint['state_dict'][k]
    model.load_state_dict(new_state_dict)

    best_module = VAEModule(model, config['exp_params'])
    best_module.freeze()

    runner.test(best_module)