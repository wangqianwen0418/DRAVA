#%%
import yaml
import torch
import numpy as np
import pandas as pd

import torch.backends.cudnn as cudnn

from models import *
from experiment import VAEModule

import shap

#%%
if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = torch.device("cpu")

def load_model(config_file, checkpoint_file):
    

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False


    model = vae_models[config['model_params']['name']](**config['model_params'])


    # load state dict from check point
    
    if torch.cuda.is_available():
        checkpoint = torch.load( checkpoint_file)
    else:
        checkpoint = torch.load( checkpoint_file, map_location=device)
    new_state_dict = {}
    for k in checkpoint['state_dict']:
        new_k = k.replace('model.', '')
        new_state_dict[new_k] = checkpoint['state_dict'][k]
    model.load_state_dict(new_state_dict)
    net = VAEModule(model, config['exp_params'])
    net.freeze()
    return net

#%%
def load_data(results_file):
    df = pd.read_csv(results_file)
    z = df['z'].apply(lambda x: [float(i) for i in x.split(',')])
    return np.array(z.to_list(), dtype = np.float32)

#%%

def calculate_shap(dataset, bg_clusters = 100, test_index = slice(2,4), nsamples=50):
    if dataset == 'matrix':
        net = load_model('./flask_server/saved_models/matrix_config.yaml', './flask_server/saved_models/matrix.ckpt')
        testing_data = load_data('../front/public/assets/results_chr1-5_10k_onTad.csv')
    elif dataset == 'celeb':
        net = load_model('./flask_server/saved_models/celeba_config.yaml', './flask_server/saved_models/celeba.ckpt')
        testing_data = load_data('../front/public/assets/results_celeba.csv')
    elif dataset == 'sequence':
        net = load_model('./flask_server/saved_models/sequence_config.yaml', './flask_server/saved_models/sequence.ckpt')
        testing_data = load_data('../front/public/assets/results_chr7_atac.csv')
    elif dataset == 'IDC':
        net = load_model('./flask_server/saved_models/IDC_config.yaml', './flask_server/saved_models/IDC.ckpt')
        testing_data = load_data('../front/public/assets/results_IDC.csv')

    def model_tobe_explained(x):
        return net.z2recons_sum(x).cpu().detach().numpy()

    explainer = shap.KernelExplainer(model_tobe_explained, shap.kmeans(testing_data, bg_clusters))

    shap_values = explainer.shap_values(testing_data[test_index], nsamples= nsamples)

    return shap_values

# %%
