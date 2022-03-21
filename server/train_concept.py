#%%
from ConceptAdaptor import *
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score
import math
 
DIM_NAME = 'sclae'
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
        # if y<=10:
        #     return 0
        # elif 10<y<=20:
        #     return 1
        # return 2
        return math.floor(y/2)

    dim_gt = 2
    dim_y = 2
    # inital acc: 0.62

else:
################
# x pos (3 classes)
################
    dim_y=7
    dim_gt=4 #posx

    def y_mapper(y):
        if y<-0.5:
            return 0
        elif y < 0.3:
            return 1
        return 2


    def gt_mapper(y):
        return math.floor(y/10)
    # inital acc: 0.70

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
def fine_tune():
    raw = np.load('./data/dsprites_test_concepts.npz')
    std = raw['std'][:, dim_y]
    n_feedback = 20
    sample_index = np.argsort(std)[::-1][:n_feedback]
    for i in range(10):
        sample_index = np.argsort(std)[::-1][:n_feedback*(i+1)]
        
        model_config = {
                "dataset": "dsprites_test_concepts",
                "data_path": "./data",
                'batch_size': 64,
                'LR': 0.005,
                'dim_y': dim_y,
                'dim_gt': dim_gt,
                'y_mapper': y_mapper,
                'gt_mapper': gt_mapper,
                'sample_index':sample_index,
                'mode': 'active'
                # 'mode': 'concept_tune'
            }

        model = ConceptAdaptor(cat_num=3, input_size=[32, 4, 4], params=model_config)

        # 'dsprites latents_names': (b'color', b'shape', b'scale', b'orientation', b'posX', b'posY')

        trainer = Trainer(gpus=0, max_epochs = 20 * (i+1), 
            early_stop_callback = False, 
            logger=False, # disable logs
            checkpoint_callback=False,
            show_progress_bar=False,
            weights_summary=None
            # reload_dataloaders_every_epoch=True # enable data loader switch between epoches
            )
        trainer.fit(model)
        trainer.test(model)

        # sample_index = model.get_uncertain_index(n_feedback*(i+1))
        # print(sample_index)


if __name__=="__main__":
    fine_tune()