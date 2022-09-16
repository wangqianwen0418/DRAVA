# Try Drava on Your Dataset

### Step 1: Train a DRL model on your dataset

- **Move your dataset into**  
  `Drava/server/data/[name]`

- **Create a data loader for your dataset**  
  `Drava/server/dataloaders.py`

- **Write a training config file**  
  `Drava/server/configs/your_config.yaml`

  <details>
  <summary>Click to View an Example Config :eyes: </summary>

  ```yaml
    model_params:
      name: 'BetaVAE'
      in_channels: 3
      latent_dim: 32 
      hidden_dims: [32, 64, 128, 256, 512]
      loss_type: 'B'
      gamma: 10.0
      max_capacity: 25
      Capacity_max_iter: 10000

    exp_params:
      dataset: celeba
      data_path: "./data/"
      img_size: 64
      optimizer: "adam"
      batch_size: 144 
      LR: 0.0005
      weight_decay: 0.0
      # scheduler_gamma: 0.95
      # if no scheduler_gama, reduce LR by factor (0.1) when a specified metric stops improving

    trainer_params:
      gpus: 1 # 0 if you want to run the model on cpu
      max_nb_epochs: 100
      max_epochs: 50

    logging_params:
      save_dir: "logs"
      name: "BetaVAE_B"
      manual_seed: 1265
    ```

  </details>

- **Start Training**
  ```
  python run.py -c [path_to_your_config_file]
  ```

### Step 2: Save the pretrained model
  - Drava will automatically save your training processes and checkpoints (based on your training config) inside `Drava/server/logs/[dataset_name]/[model_name]/[version]`.
  
  - After training is finished, you can select the version that you are most stasfied. You can then move the checkpoint file (`[name].ckpt`), the corresponding training config file (`[name]_config.yaml`), and  the latent vector file (`results_[name].csv`) into `Drava/server/flask_server/saved_model`. 
  - If you have custom dimensions or item labels that you want to add, please add them into the `result_[name].csv` as additional columns.

### Step 3: Run the front- and back-end server
 Run the front-end and back-end server [as instructed](./dev.md).

### Step 4: Add entry points in the front-end
  - Specify you dataset in `front/src/config/dataset_config.ts` by adding a key-value pair to the `datasetConfig` variable. The added dataset config should include the name of the dataset, the views you want to include, the item labels from metadata, and other user-defined dimensions. You should use the same `[name]` as you used in the checkpoint file, the latent vector file, and the train config file.

  
  ```javascript
    [name]: {
      name: 'CelebA',

      // define the layout of views, Drava currenlty supports four types of views: `latentDim`, `itemView`, `contextView`, `gosling`
      views: { left: ['latentDim'], right: ['itemView'] },

      // specify the labels used in the item broweser. 
      // Label names should be the same as the column names in the results_[name].csv file
      labels: ['gender', 'smiling', 'hair', 'bangs', 'young'],

      // specify the custom dimensions to be added in the Concept View. 
      // Dimension names should be the same as the column names in the results_[name].csv file
      customDims: ['recons_loss'],
    },
  ```
    

### Step 5: (Optional) custom APIs
- By default, we assume each data item is an image stored at `Drava/data/[name]/[id]`. If you want to use other data formats that requires different loading functions, you can specify that in `Drava/server/flask_server/api.py`

  Below is an example. `[name]` is the same name your specified in previous steps.
  ```python
  def get_[name]_sample(id):
      id = request.args.get('id', type=str)
      item = # write your loading functions here
      return item
  ```

### Step 6: Interact with your model and data in Drava
   Now you can open `localhost:3000` in your web browser and interact with your dataset :tada:  
   Feel free to [open an github issue](https://github.com/wangqianwen0418/DRAVA/issues/new/choose) if you run into any problem!