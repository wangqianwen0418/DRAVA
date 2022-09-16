#%%
from ConceptAdaptor import *
from pytorch_lightning import Trainer
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks import EarlyStopping
import shutil

classes = ['5_o_Clock_Shadow',
 'Arched_Eyebrows',
 'Attractive',
 'Bags_Under_Eyes',
 'Bald',
 'Bangs',
 'Big_Lips',
 'Big_Nose',
 'Black_Hair',
 'Blond_Hair',
 'Blurry',
 'Brown_Hair',
 'Bushy_Eyebrows',
 'Chubby',
 'Double_Chin',
 'Eyeglasses',
 'Goatee',
 'Gray_Hair',
 'Heavy_Makeup',
 'High_Cheekbones',
 'Male',
 'Mouth_Slightly_Open',
 'Mustache',
 'Narrow_Eyes',
 'No_Beard',
 'Oval_Face',
 'Pale_Skin',
 'Pointy_Nose',
 'Receding_Hairline',
 'Rosy_Cheeks',
 'Sideburns',
 'Smiling',
 'Straight_Hair',
 'Wavy_Hair',
 'Wearing_Earrings',
 'Wearing_Hat',
 'Wearing_Lipstick',
 'Wearing_Necklace',
 'Wearing_Necktie',
 'Young']
#%%

# dim = 'bangs'
dim = 'smiling'
# 

if dim == 'smiling':
    # mappers for smiling
    def y_mapper(y):
        if y<0.1:
            return 0
        return 1


    def gt_mapper(y):
        return y

    # smiling
    dim_y = 19
    dim_gt = 31

else:
    # #########
    # bangs: 0.77
    ##############
    def y_mapper(y):
        if y<-0.6:
            return 1
        return 0


    def gt_mapper(y):
        return y

    # bangs: 
    dim_y = 10
    dim_gt = 5
# %%
# intial accuracy
def initial_acc():
    raw = np.load('./data/celeba_concepts.npz')
    y_true = raw['gt'][:, dim_gt] 
    y_true_norm = np.vectorize(gt_mapper)(y_true)

    y_pred = raw['y'][:, dim_y]
    y_pred_norm = np.vectorize(y_mapper)(y_pred)

    acc = accuracy_score(y_true_norm, y_pred_norm, normalize=True)
    print('initial', acc)


# std = raw['std'][:, dim]
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
def fine_tune(sample, mode='concept_tune'):
    raw = np.load('./data/celeba_concepts.npz')
    y_true = raw['gt'][:, dim_gt]
    y_true_norm = np.vectorize(gt_mapper)(y_true)

    y_pred = raw['y'][:, dim_y]
    y_pred_norm = np.vectorize(y_mapper)(y_pred)
    std = raw['std'][:, dim_y]

    # n_feedback = int(0.01 * len(raw['gt']))
    n_feedback = int(0.02 * len(raw['gt']))
    # n_feedback = int(0.05 * len(raw['gt']))

    n_first = int(0.05 * len(raw['gt']))
    sample_index = np.argsort(std)[::-1][:n_first]
    for i in range(15):
        
        model_config = {
                "dataset": "celeba_concepts",
                "data_path": "./data",
                'batch_size': 64,
                'LR': 0.005,
                'dim_y': dim_y,
                'dim_gt': dim_gt,
                'y_mapper': y_mapper,
                'gt_mapper': gt_mapper,
                'sample_index': np.array([]) if mode=='concept_tune' and i==0 else sample_index,
                # 'mode': 'active'
                'mode': mode
            }

        model = ConceptAdaptor(cat_num=3, input_size=[512, 2, 2], params=model_config)

        # 'dsprites latents_names': (b'color', b'shape', b'scale', b'orientation', b'posX', b'posY')

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
        if sample == 'uncertain' or model_config['mode']=='active':
            sample_index = model.get_uncertain_index(n_feedback*(i+1))
        #  based on std
        elif sample == 'std':
            sample_index = np.argsort(std)[::-1][:n_feedback*(i+1)]
        # based on mean value
        elif sample == 'm':
            sample_index = [ i for i in np.argsort( np.abs(y_pred -( -0.6)) ) if y_pred_norm[i]!=y_true_norm[i]][:n_feedback*(i+1)]
            


if __name__=="__main__":
    if os.path.exists('lightning_logs/'):
        shutil.rmtree('lightning_logs/', ignore_errors=True)
    fine_tune(sample = 'uncertain', mode='active')
# %%
