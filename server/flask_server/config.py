import os
# from model_loader_static import ModelLoader
SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    FRONT_ROOT = os.path.join(SERVER_ROOT, 'build')
    DATA_FOLDER = os.path.join(SERVER_ROOT, 'build/assets')
    STATIC_FOLDER = os.path.join(SERVER_ROOT, 'build/static')
    BIN_NUM = 11
