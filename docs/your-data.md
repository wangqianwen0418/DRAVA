# Try Drava on Your Dataset

### Step 0: Run the front- and back-end server
 Run the [front-end](./front-end.md) and [back-end](./back-end.md) server as instructued.

### Step 1: Train a DRL model on your dataset

- **Move your dataset into**  
  `Drava/server/data`

- **Create a data loader for your dataset**  
  `Drava/server/dataloaders.py`

- **Write a training config file**  
  `Drava/server/configs/your_config.yaml`

- **Start Training**
  ```
  python run.py -c [path_to_your_config_file]
  ```

### Step 2: Save the pretrained model
  - Drava will automatically save your training processes and checkpoints (based on your training config) inside `Drava/server/logs/[dataset_name]/[model_name]/[version]`.
  
  - After training is finished, you can select the version that you are most stasfied. You can then move the checkpoint file (`[name].ckpt`) and the corresponding training config file (`[name]_config.yaml`) into `Drava/server/flask_server/saved_model`. And the latent vector file (`results_[name].csv`) into `Drava/front/public/assets`

### Step 3: Add entry points in the front-end
  - Specify you dataset in `front/src/config/dataset_config.ts` by adding a key-value pair to the `datasetConfig` variable. The added dataset config should include the name of the dataset, the views you want to include, the item labels from metadata, and other user-defined dimensions. The key should should be the `[name]` you used in the checkpoint file, the latent vector file, and the train config file.

    For example:
    ```
    [name]: {
      name: 'CelebA',
      labels: ['gender', 'smiling', 'hair', 'bangs', 'young'],
      customDims: ['recons_loss'],
      views: { left: ['latentDim'], right: ['itemView'] }
    },
    ```

### Step 4: (Optional) custom APIs
- By default, we assume each data item is an image stored at `Drava/data/[name]/[id]`. If you want to use other data formats that requires different loading functions, you can specify that in `Drava/server/flask_server/api.py`

  Below is an example. `[name]` is the same name your specified in Step 2 and 3.
  ```python
  def get_[name]_sample(id):
      id = request.args.get('id', type=str)
      item = # write your loading functions here
      return item
  ```

### Step 5: Interact with your model and data in Drava
   Now you can open `localhost:3000` in your web browser and interact with your dataset :tada:
   Feel free to [open an github issue](https://github.com/wangqianwen0418/DRAVA/issues/new/choose) if you run into any problem :)