import json
import numpy as np

import flask
from flask import request, jsonify, Blueprint, current_app, g

api = Blueprint('api', __name__)
######################
# API Starts here
######################


@api.route('/test', methods=['GET'])
def test():
    return 'api test successfully'

@api.route('/get_matrix_sample', methods=['GET'])
def get_matrix_sample(filename:str, chr:str, start:int, end:int):
    '''
    e.g, base_url/api/?chr=1&start=0&end=1000
    '''
    return 


######################
# functions called by the API
######################



def get_sequence_sample(filename:str, chr:str, start:int, end:int):
    return 

def get_image_sample(filename:str):
    return


def getReconstructedSample(model_name:str, vector: list):
    return 