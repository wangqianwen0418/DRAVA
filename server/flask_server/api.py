# %%
# 
# %% 
from crypt import methods
import json
from operator import mod
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from matplotlib import cm as colormap
from matplotlib.colors import LinearSegmentedColormap
import torchvision.utils as vutils

import zarr
import os

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
    t.add_(-min).div_(max - min + 1e-5)  # add 1e-5 in case min = max


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

    model = vae_models[config['model_params']
                       ['name']](**config['model_params'])

    # load state dict from check point

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    new_state_dict = {}
    for k in checkpoint['state_dict']:
        new_k = k.replace('model.', '')
        new_state_dict[new_k] = checkpoint['state_dict'][k]
    model.load_state_dict(new_state_dict)
    net = VAEModule(model, config['exp_params'])
    net.freeze()
    return net

models = {}
ranges = {}

default_z = {
    "dsprites": [-0.027196992188692093, 0.062033019959926605, 1.2151720523834229, -0.7173954248428345, 2.0358076095581055, -0.004620308056473732, 0.031831007450819016, 1.2410718202590942, -0.0464935339987278, 0.03360012546181679],
    "IDC": [-0.07878086715936661, 0.6129785776138306, 1.6250171661376953, -0.26838210225105286, 2.1170804500579834, 0.7363924384117126, 0.07876656204462051, -0.36886027455329895, 0.017318010330200195, -0.9062463045120239, -0.2743624746799469, -1.159773349761963],
    'sequence': [1.3912068605422974, 1.3093589544296265, -1.4369394779205322, 2.921229362487793, 1.7272869348526, -1.0809800624847412],
    'matrix': [0.5784032344818115, 0.1713341921567917, -0.27981624007225037, -0.4180270731449127, 0.9767476916313171, -0.7862354516983032, 0.7032433152198792, 0.7099565863609314],
    "sc2": [-0.17151187360286713,-0.1341707855463028,0.11569507420063019,-0.2921229898929596,-0.038597069680690765,0.18862436711788177,-0.05055750161409378,0.006634707096964121,0.2749462425708771,-2.4682295322418213,-1.207783818244934,-0.9922627210617065,0.11990928649902344,0.4271310865879059,-0.5906530618667603,-0.01368972472846508,-0.08378960937261581,0.053271450102329254,-1.072157621383667,-0.030002571642398834,0.18833142518997192,-0.07939216494560242,-0.1480628252029419,0.5668376684188843,0.0033783912658691406],
    'celeba': [0.35241496562957764, -1.446256399154663, 0.6035149097442627, -1.4706382751464844, -0.6200129389762878, -0.44358429312705994, -1.6820268630981445, 1.6138064861297607, -1.7537750005722046, -1.098387360572815, 0.27564120292663574, 2.4112865924835205, -0.7761713266372681, 0.500797688961029, 1.3642232418060303, 1.607535719871521, -0.0050630271434783936, 0.21523889899253845, -0.5679569244384766, -0.4611191749572754]
}

sequence_data = np.load(
    safe_join('../data/', 'HFFc6_ATAC_chr7.npz'), encoding='bytes')['imgs']
dsprites_data = np.load(
    safe_join('../data/', 'dsprites_test.npz'), encoding='bytes')['imgs']
sc2_data = zarr.open(f'../data/codex/HBM622.JXWQ.554/cell_patches_2cluster.zarr', mode='r')
######################
# API Starts here
######################


@api.route('/test', methods=['GET'])
def test():
    return 'api test successfully'

@api.route('/get_item_sample', methods=['GET'])
def get_item_sample():
    '''
    e.g., base_url/api/get_matrix_sample?id=xx&dataset=xxx
    '''
    id = request.args.get('id', type=str)
    dataset = request.args.get('dataset', type=str)

    try:
        # call function name based on variable
        return globals()[f'get_{dataset}_sample'](id)
    except Exception:
        print(Exception)
        return send_file(f'../data/{dataset}/{id}')

@api.route('/get_simu_images', methods=['GET'])
def get_simu_images():
    '''
    :param dataset: name of dataset
    :param dim: index of dimension
    :param z: the latent vector used to generate simu images. if not specified, return default from the simu folder
    :return a list of images of byte array 
    e.g., base_url/api/get_simu_images?dataset=matrix&dim=2&z='0.2,0.3,-0.2,-0.3'
    '''
    global models, ranges

    BIN_NUM = current_app.config['BIN_NUM']
    dim = request.args.get('dim', type=int)
    dataset = request.args.get('dataset', type=str)
    z = request.args.get('z', type=str)

    if z:
        z = [float(i) for i in z.split(',')]
    elif dataset in default_z:
        z = default_z[dataset]
    else:
        z = []

    if dataset in ranges:
        if os.path.exists(f'saved_models/z_range_{dataset}.json'):
            with open('saved_models/z_range_sc2.json', 'r') as f:
                ranges[dataset] = json.load(f)
        zRange = ranges[dataset][dim]
    else:
        zRange = [-3, 3]

    if not dataset in models:
        models[dataset] = load_model(f'./saved_models/{dataset}_config.yaml', f'./saved_models/{dataset}.ckpt')

    reconstructued, score = models[dataset].get_simu_images(dim, z, zRange)

    results = []

    for res in reconstructued:

        img_io = BytesIO()

        # =========reshape numpy array=========
        if (dataset == 'celeba' or dataset == 'IDC'):
            # image shape from [3, 64, 64] to [64, 64, 3]
            res = np.rollaxis(res, 0, 3)
        elif dataset == 'sc2':
            res = np.argmax(res, axis=0) # each pixel from one hot vector to class index
        else:
            res = res[0]  # image shape from [1, 64, 64] to [64, 64]

        # ==========numpy array to pil image object=========
        if dataset == 'matrix':  # changef from grayscale to a defined color map
            res = colormap.get_cmap('viridis')(res) * 255
            pil_img = Image.fromarray(res.astype(np.uint8)).convert('RGB')
        elif dataset == 'sc2':
            pil_img = cate_arr_to_image(res)
        else:
            if (dataset) == 'dsprites':
                res = 1-res
            res = res*255
            res = res.astype(np.uint8)
            pil_img = Image.fromarray(res)

        # ===================
        pil_img.save(img_io, 'png', quality=100)
        img_io.seek(0)
        v = base64.b64encode(img_io.getvalue()).decode()
        results.append(f'data:image/png;base64,{v}')

    # a quick hack for the y pos axis
    # TODO: enable users to reverse an axis
    if dataset == 'dsprites' and dim == 4:
        results = results[::-1]

    return jsonify({"image": results, "score": score})


@api.route('/get_model_results', methods=['GET'])
def get_model_results():
    '''
    :param dataset: name of dataset
    :return: Array<{[key:string]: value}>
    e.g., base_url/api/get_model_results?dataset=matrix
    '''
    dataset = request.args.get('dataset', type=str)

    try:
        # call function name based on variable
        return globals()[f'get_{dataset}_results']()
    except Exception:
        return get_default_results(dataset)

######################
# functions called by the API
######################
def cate_arr_to_image(arr, border=False):
    '''
    converting a categorical mask to an image
    each value in array indicates a class index.
    pixels belonging to the same class will be coded using the same color

    :param arr: numpy array, shape [h, w]
    :param border: boolean, whether to add a border to the returned image
    :return: an image
    '''
    colors = [(1, 1, 1), (1, 0.5, 0) , (0, 0.7, 0)] # white (bg), red(cell), green (nucleus)
    mycolormap = LinearSegmentedColormap.from_list('myCmap', colors, N=3)
    res = mycolormap(arr) * 255

    # # add a border
    if border:
        res[0, :, :3] = 50
        res[62, :, :3] = 50
        res[:, 0, :3] = 50
        res[:, 62, :3] = 50

        pil_img = Image.fromarray(res.astype(np.uint8)).convert('RGBA')
    
    else:
        # transparent background
        newData = []
        pil_img = Image.fromarray(res.astype(np.uint8))

        for item in pil_img.getdata():
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        
        pil_img.putdata(newData)
    
    return pil_img

########### get data item for different dataset
def get_matrix_sample(id):
    img_src = Image.open(f'../data/tad_imgs/chr5:{int(id)}.jpg').convert('L')
    im = np.array(img_src)
    im = colormap.get_cmap('viridis')(im) * 255
    pil_img = Image.fromarray(im.astype(np.uint8)).convert('RGB')
    pil_img = pil_img.resize((64, 64), Image.NEAREST)

    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

def get_IDC_sample(id):
    return send_file(f'../data/IDC_regular_ps50_idx5/{id}')


def get_sequence_sample(id):
    img = sequence_data[int(id)]*255
    # add a border
    img[0, :] = 50
    img[62, :] = 50
    img[:, 0] = 50
    img[:, 62] = 50
    pil_img = Image.fromarray(img.astype(np.uint8))

    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def get_dsprites_sample(id):
    '''
    e.g., base_url/api/get_dsprites_sample?id=xx
    '''
    id = request.args.get('id', type=str)
    img = dsprites_data[int(id)]*255
    # convert white to black
    img = 255 - img
    # add a border
    img[0, :] = 50
    img[62, :] = 50
    img[:, 0] = 50
    img[:, 62] = 50
    pil_img = Image.fromarray(img.astype(np.uint8))

    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def get_celeba_sample(id):
    return send_from_directory(f'../data/celeba/img_align_celeba/', f'{int(id):06}.jpg')

def get_sc2_sample(id):
    res = sc2_data[int(id)]
    pil_img = cate_arr_to_image(res)
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

############### get results for different datasets
def parse_results(df):

    df['z'] = df['z'].apply(lambda x: [ float(i) for i in x.split(',')])
    df['embedding'] = df['z'].apply(lambda x: [ float(i)/6+0.5 for i in x])

    for i in range(len(df['z'][0])):
        df[f'dim_{i}'] = df['z'].apply(lambda x: x[i])
    df['index'] = df.index
    df['id'] = df['index'].apply(lambda x: str(x+1))

    return df

def return_results(df):
    '''
    pandas df to a json array
    '''
    res = json.loads(df.to_json(orient='records'))
    for i, r in enumerate(res):
        r['assignments'] = {}

    return jsonify(res)

def get_default_results(dataset):
    url = f'./saved_models/results_{dataset}.csv'
    df = pd.read_csv(url)
    return return_results( parse_results(df) )

def get_matrix_results():
    url = './saved_models/results_chr1-5_10k_onTad.csv'
    resolution = 10000 # 10k

    df = pd.read_csv(url)
    df = df[df['chr'] == 5]

    df = parse_results(df)
    df['start'] = df['start'] * resolution
    df['end'] = df['end'] * resolution
    df['size'] = df['end'] - df['start']
    

    return return_results(df)

def get_dsprites_results():
    url = f'./saved_models/results_dsprites.csv'
    df = pd.read_csv(url)
    df = parse_results(df)
    df['id'] = df['index'].apply(str)
    df['dim_4'] = -1*df['dim_4']
    df['embedding'] = df['embedding'].apply(lambda z: [z[i] for i in [2, 3, 4, 7, 0]])
    return return_results( df )

def get_IDC_results():
    url = f'./saved_models/results_IDC.csv'
    df = pd.read_csv(url)
    df = parse_results(df)

    df = df.drop(columns=['id'])
    df = df.rename(columns={'acc': 'prediction', 'img_path': 'id'})
    df['prediction'] = df['prediction'].apply(lambda x: 'pos' if x else 'neg')
    df['label'] = df['label'].apply(lambda x: 'pos' if x==1 else 'neg')
    return return_results( df )
