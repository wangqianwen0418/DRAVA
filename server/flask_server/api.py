from crypt import methods
import json
from operator import mod
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from matplotlib import cm as colormap
import torchvision.utils as vutils

import flask
from flask import request, jsonify, safe_join, send_from_directory, send_file, Blueprint, current_app, g

# 
import torch

api = Blueprint('api', __name__)


######################
# load model
######################

import yaml
import torch.backends.cudnn as cudnn
import sys
sys.path.append('../../server')
from models import *
from experiment import VAEModule

def norm_range(t):
    min = float(t.min())
    max = float(t.max())
    t.clamp_(min=min, max=max)
    t.add_(-min).div_(max - min + 1e-5) # add 1e-5 in case min = max

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

matrix_model = load_model('./saved_models/matrix_config.yaml', './saved_models/matrix.ckpt')
celeba_model = load_model('./saved_models/celeba_config.yaml', './saved_models/celeba.ckpt')
sequence_model = load_model('./saved_models/sequence_config.yaml', './saved_models/sequence.ckpt')
IDC_model = load_model('./saved_models/IDC_config.yaml', './saved_models/IDC.ckpt')
dsprites_model = load_model('./saved_models/dsprites_config.yaml', './saved_models/dsprites.ckpt')

with open('saved_models/z_range_matrix.json', 'r') as f:
    range_matrix = json.load(f)

with open('saved_models/z_range_sequence.json', 'r') as f:
    range_sequence = json.load(f)

with open('saved_models/z_range_celeba.json', 'r') as f:
    range_celeba = json.load(f)

with open('saved_models/z_range_IDC.json', 'r') as f:
    range_IDC = json.load(f)

with open('saved_models/z_range_dsprites.json', 'r') as f:
    range_dsprites = json.load(f)

ranges = {
    'sequence': range_sequence,
    'matrix': range_matrix,
    # 'celeb': range_celeba,
    'IDC': range_IDC,
    # 'dsprites': range_dsprites
}

models = {
    'sequence': sequence_model,
    'matrix': matrix_model,
    'celeb': celeba_model,
    'IDC': IDC_model,
    'dsprites': dsprites_model
}

default_z = {
    "dsprites": [-0.027196992188692093,0.062033019959926605,1.2151720523834229,-0.7173954248428345,2.0358076095581055,-0.004620308056473732,0.031831007450819016,1.2410718202590942,-0.0464935339987278,0.03360012546181679],
    "IDC": [-0.07878086715936661,0.6129785776138306,1.6250171661376953,-0.26838210225105286,2.1170804500579834,0.7363924384117126,0.07876656204462051,-0.36886027455329895,0.017318010330200195,-0.9062463045120239,-0.2743624746799469,-1.159773349761963],
    'sequence': [1.3912068605422974,1.3093589544296265,-1.4369394779205322,2.921229362487793,1.7272869348526,-1.0809800624847412],
    'matrix': [0.5784032344818115,0.1713341921567917,-0.27981624007225037,-0.4180270731449127,0.9767476916313171,-0.7862354516983032,0.7032433152198792,0.7099565863609314],
    'celeb': [0.35241496562957764,-1.446256399154663,0.6035149097442627,-1.4706382751464844,-0.6200129389762878,-0.44358429312705994,-1.6820268630981445,1.6138064861297607,-1.7537750005722046,-1.098387360572815,0.27564120292663574,2.4112865924835205,-0.7761713266372681,0.500797688961029,1.3642232418060303,1.607535719871521,-0.0050630271434783936,0.21523889899253845,-0.5679569244384766,-0.4611191749572754]
}

sequence_data = np.load(safe_join('../data/', 'HFFc6_ATAC_chr7.npz'), encoding='bytes')['imgs']
dsprites_data = np.load(safe_join('../data/', 'dsprites_test.npz'), encoding='bytes')['imgs']
######################
# API Starts here
######################




@api.route('/test', methods=['GET'])
def test():
    return 'api test successfully'

# @api.route('/get_matrix_sample', methods=['GET'])
# def get_matrix_sample():
#     '''
#     e.g., base_url/api/get_matrix_sample?id=xx
#     '''
#     id = request.args.get('id', type=str)
#     return send_from_directory(safe_join('../data/', 'tad_imgs'), f'chr5:{int(id)+1}.jpg')

@api.route('/get_matrix_sample', methods=['GET'])
def get_matrix_sample():
    '''
    e.g., base_url/api/get_matrix_sample?id=xx
    '''
    id = request.args.get('id', type=str)
    img_src = Image.open(f'../data/tad_imgs/chr5:{int(id)}.jpg').convert('L')
    im = np.array(img_src)
    im = colormap.get_cmap('viridis')(im) * 255
    pil_img = Image.fromarray(im.astype(np.uint8)).convert('RGB')

    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpg')

@api.route('/get_IDC_sample', methods=['GET'])
def get_IDC_sample():
    '''
    e.g., base_url/api/get_IDC_sample?id=xx
    '''
    id = request.args.get('id', type=str)
    return send_file(f'../data/IDC_regular_ps50_idx5/{id}')
    

@api.route('/get_sequence_sample', methods=['GET'])
def get_sequence_sample():
    '''
    e.g., base_url/api/get_sequence_sample?id=xx
    '''
    id = request.args.get('id', type=str)
    img = sequence_data[int(id)]*255
    # add a border
    img[0, :] = 20
    img[63, :] = 20
    img[:, 0] = 20
    img[:, 63] = 20
    pil_img = Image.fromarray(img.astype(np.uint8))
    
    
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpg')

@api.route('/get_dsprites_sample', methods=['GET'])
def get_dsprites_sample():
    '''
    e.g., base_url/api/get_dsprites_sample?id=xx
    '''
    id = request.args.get('id', type=str)
    img = dsprites_data[int(id)]*255
    # convert white to black
    img = 255- img
    # add a border
    img[0, :] = 20
    img[63, :] = 20
    img[:, 0] = 20
    img[:, 63] = 20
    pil_img = Image.fromarray(img.astype(np.uint8))
    
    
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpg')


@api.route('/get_celeb_sample', methods=['GET'])
def get_celeb_sample():
    '''
    e.g., base_url/api/get_celeb_sample?id=xx
    '''
    dataset = 'celeba'

    id = request.args.get('id', type=str)
    return send_from_directory(f'../data/{dataset}/img_align_celeba/', f'{int(id):06}.jpg')

@api.route('/get_simu_images', methods=['GET'])
def get_simu_images():
    '''
    :param dataset: name of dataset
    :param dim: index of dimension
    :param z: the latent vector used to generate simu images. if not specified, return default from the simu folder
    :return a list of images of byte array 
    e.g., base_url/api/get_simu_images?dataset=matrix&dim=2&z='0.2,0.3,-0.2,-0.3'
    '''

    BIN_NUM = current_app.config['BIN_NUM']
    dim = request.args.get('dim', type=int)
    dataset = request.args.get('dataset', type=str)
    z=request.args.get('z', type=str)

    if z:
        z= [float(i) for i in z.split(',')]
    else:
        z = default_z[dataset]

    if dataset in ranges:
        zRange = ranges[dataset][dim]
    else:
        zRange = [-3, 3]

    reconstructued, score = models[dataset].get_simu_images(dim, z, zRange)

    results = []

    for res in reconstructued:

        img_io = BytesIO()
       
        if (dataset == 'celeb' or dataset == 'IDC'):
            res = np.rollaxis(res,0,3) # image shape from [3, 64, 64] to [64, 64, 3]
        else:
            res = res[0] # image shape from [1, 64, 64] to [64, 64]

        if dataset == 'matrix': # changef from grayscale to a defined color map
            res = colormap.get_cmap('viridis')(res) * 255
            pil_img = Image.fromarray(res.astype(np.uint8)).convert('RGB')
        else:
            if (dataset) == 'dsprites':
                res = 1-res
            res = res*255
            res = res.astype(np.uint8)
            pil_img = Image.fromarray(res)
        pil_img.save(img_io, 'png', quality=100)
        img_io.seek(0)
        v = base64.b64encode(img_io.getvalue()).decode()
        results.append(f'data:image/png;base64,{v}')

    return jsonify({"image": results, "score": score})

######################
# functions called by the API
######################

