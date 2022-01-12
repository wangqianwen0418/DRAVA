import os
# from model_loader_static import ModelLoader
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    FRONT_ROOT = os.path.join(SERVER_ROOT, 'build')
    DATA_FOLDER = os.path.join(SERVER_ROOT, 'data')
    STATIC_FOLDER = os.path.join(SERVER_ROOT, 'build/static')
    GNN = 'attention'
    # MODEL_LOADER = ModelLoader(os.path.join(SERVER_ROOT, 'collab_delivery/'))
    # MODEL_LOADER = ModelLoader('s3://drug-gnn-models/collaboration_delivery/')
