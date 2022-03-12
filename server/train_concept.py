#%%
from ConceptAdaptor import *
from pytorch_lightning import Trainer

#%%


y_mapper = lambda y: 1
gt_mapper = lambda y: y

model_config = {
        "dataset": "dsprites_test_concepts",
        "data_path": "./data",
        'batch_size': 64,
        'LR': 0.002,
        'dim': 1,
        'y_mapper': y_mapper,
        'gt_mapper': gt_mapper
    }

model = ConceptAdaptor(cat_num=3, input_size=[32, 4, 4], params=model_config)

# 'dsprites latents_names': (b'color', b'shape', b'scale', b'orientation', b'posX', b'posY')

trainer = Trainer(gpus=0)
trainer.fit(model)
# %%
