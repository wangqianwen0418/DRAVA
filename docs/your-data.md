# Try Drava on Your Dataset

### Step 0: Run the front- and back-end server

### Step 1: Train a DRL model on your dataset

- **Create a data loader for your dataset**  
  `Drava/server/dataloaders.py`

- **Write a training config file**  
  `Drava/server/configs/your_config.yaml`

- **Start Training**
  ```
  python run.py -c [path_to_your_config_file]
  ```

### Step 2: Save the pretrained model
  Drava will automatically save your training processes and checkpoints (based on your training config) inside `Drava/server/logs`.
  
  After training is finished, please move one checkpoint file and the training config file into `Drava/server/flask_server/saved_model`


### Step 3: Add entry points in the front-end

### Step 4: Interact with your model and data in Drava