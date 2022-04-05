# DRAVA: Utilizing **D**isentangled **R**epresentation Learning as **A** **V**isual **A**nalytics method for pattern-based data exploration

This repository has two main components: a frontend interface and a back-end server.

## Development

### Backend

First, install all dependent packages:

```sh
cd server
pip install -r requirements.txt
```

Then, start the flask server:

```sh
cd flask_server
python app.py
```

To manage dependencies more effectively, you can create and use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) before installing all packages:

```sh
conda create -n drava
conda activate drava
# ...
conda deactivate
```

The pre-trained models are stored at `server/flask_server/saved_models`.

### Frontend

First, install all dependent packages:

```sh
cd front
npm install
```

Then, launch the Drava react application on the browser:

```sh
npm start
```

## Datasets Used
- https://www.kaggle.com/paultimothymooney/breast-histopathology-images
- https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ 