import json
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib import cm as colormap

import flask
from flask import request, jsonify, safe_join, send_from_directory, send_file, Blueprint, current_app, g

api = Blueprint('api', __name__)
######################
# API Starts here
######################

sequence_data = []


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
    img_src = Image.open(f'../data/tad_imgs/chr5:{int(id)+1}.jpg').convert('L')
    im = np.array(img_src)
    im = colormap.get_cmap('viridis')(im) * 255
    pil_img = Image.fromarray(im.astype(np.uint8)).convert('RGB')

    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpg')
    

@api.route('/get_sequence_sample', methods=['GET'])
def get_sequence_sample():
    '''
    e.g., base_url/api/get_sequence_sample?id=xx
    '''
    global sequence_data
    dataset = 'HFFc6_ATAC_chr7.npz'
    if len(sequence_data) == 0:
        print('reload npz')
        sequence_data = np.load(safe_join('../data/', dataset), encoding='bytes')['imgs']
    id = request.args.get('id', type=str)
    img = sequence_data[int(id)]*255
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
    print(f'{int(id):06}.jpg')
    return send_from_directory(f'../data/{dataset}/img_align_celeba/', f'{int(id)+1:06}.jpg')


######################
# functions called by the API
######################





def get_image_sample(filename:str):
    return


def getReconstructedSample(model_name:str, vector: list):
    return 