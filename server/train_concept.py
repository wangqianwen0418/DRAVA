#%%
from ConceptAdaptor import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import math
import shutil

# 'dsprites latents_names': (b'color', b'shape', b'scale', b'orientation', b'posX', b'posY')
DIM_NAME = 'scale'
#%%
if DIM_NAME == 'scale':
    ################
    # scale (3 classes)
    ################
    # mappers for dsprites scale
    def y_mapper(y):
        if y<-0.9:
            return 2
        elif y < 0.4:
            return 1
        return 0


    def gt_mapper(y):
        return math.floor(y/2)

    dim_gt = 2
    dim_y = 2
    # inital acc: 0.62

elif DIM_NAME == 'pos_x':
################
# x pos (3 classes)
################
    dim_y=7
    dim_gt=4 #posx

    def y_mapper(y):
        if y<-0.5:
            return 0
        elif y < 0.4:
            return 1
        return 2


    def gt_mapper(y):
        if y<=10:
            return 0
        elif 10<y<=20:
            return 1
        return 2
    # inital acc: 0.871

elif DIM_NAME == 'pos_y':
################
# y pos (3 classes)
################
    dim_y= 4
    dim_gt=5 #pos y

    def y_mapper(y):
        if y<-0.5:
            return 0
        elif y < 0.7:
            return 1
        return 2


    def gt_mapper(y):
        if y<=10:
            return 0
        elif 10<y<=20:
            return 1
        return 2
    #  acc 0.93
# %%
# intial accuracy
def initial_acc():
    raw = np.load('./data/dsprites_test_concepts.npz')
    y_true = raw['gt'][:, dim_gt]
    y_true_norm = np.vectorize(gt_mapper)(y_true)

    y_pred = raw['y'][:, dim_y]
    y_pred_norm = np.vectorize(y_mapper)(y_pred)

    acc = accuracy_score(y_true_norm, y_pred_norm, normalize=True)
    print('initial', acc)


# std = raw['std'][:, dim_y]
# sample_index = np.argsort(std)[::-1]
# n_group = 10
# for i in range(n_group):
#     index = sample_index[int(i*len(std)/n_group): int((i+1)*len(std)/n_group)] 
#     y1 = y_pred_norm[index]
#     y2 = y_true_norm[index]
#     acc= accuracy_score(y1, y2, normalize=True)
#     print(i, acc)

#%%
#  model training
def fine_tune(sample='m', mode='concept_tune'):
    raw = np.load('./data/dsprites_test_concepts.npz')
    std = raw['std'][:, dim_y]
    y_true = raw['gt'][:, dim_gt]
    y_true_norm = np.vectorize(gt_mapper)(y_true)

    y_pred = raw['y'][:, dim_y]
    y_pred_norm = np.vectorize(y_mapper)(y_pred)

    # n_feedback = int(0.01 * len(raw['gt']))
    # n_feedback = int(0.02 * len(raw['gt']))
    n_feedback = int(0.05 * len(raw['gt']))
    n_first = int(0.05 * len(raw['gt']))
    sample_index = np.argsort(std)[::-1][:n_first]
    for i in range(15):
        
        
        model_config = {
                "dataset": "dsprites_test_concepts",
                "data_path": "./data",
                'batch_size': 64,
                'LR': 0.005,
                'dim_y': dim_y,
                'dim_gt': dim_gt,
                'y_mapper': y_mapper,
                'gt_mapper': gt_mapper,
                'sample_index': np.array([]) if mode=='concept_tune' and i==0 else sample_index,
                # 'mode': 'active'
                # 'mode': 'concept_tune'
                'mode': mode
            }


        model = ConceptAdaptor(cat_num=3, input_size=[32, 4, 4], params=model_config)

        trainer = Trainer(gpus=0, max_epochs = 100, 
            early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False), 
            logger=True,
            # the default checkpoint callback will restore the model from the last checkpoint
            show_progress_bar=False,
            weights_summary=None
            )
        trainer.fit(model)
        trainer.test(model)

        # based on probability score
        if sample == 'uncertain':
            sample_index = model.get_uncertain_index(n_feedback*i+n_first)
        #  based on std
        elif sample == 'std':
            sample_index = np.argsort(std)[::-1][:n_feedback*i+n_first]
        # based on mean value
        elif sample == 'm':
            sample_index_a = [ i for i in np.argsort( np.abs(y_pred - 0.4) ) if y_pred_norm[i]!=y_true_norm[i]][:int((n_feedback*i+n_first)/2)]
            sample_index_b = [i for i  in np.argsort( np.abs(y_pred - (-0.9)) ) if y_pred_norm[i]!=y_true_norm[i]][:int((n_feedback*i+n_first)/2)]
            sample_index = np.concatenate((sample_index_a, sample_index_b))
        # print(sample_index)

#%%
if __name__=="__main__":
    if os.path.exists('lightning_logs/'):
        shutil.rmtree('lightning_logs/', ignore_errors=True)
    fine_tune(sample='m', mode='concept_tune')
# %%
