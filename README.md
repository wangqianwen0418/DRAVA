# DRAVA: Utilizing <ins>D</ins>isentangled <ins>R</ins>epresentation Learning as <ins>A</ins> <ins>V</ins>isual <ins>A</ins>nalytics Method for Pattern-based Data Exploration

Check out our preprint at https://osf.io/gzu27/

This repository has two main components: a frontend interface and a back-end server.

## Development

Please refer to the [documentation webpage](https://qianwen.info/DRAVA) for more details

### Run the doc webpage

```
docsify serve docs
```

### Backend
The backend is developed and tested with `python@3.7.9`

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

### Required Datasets

To run both the server and the client, you need to put additional files into your local repository. These include

- `server/data/` (Image patches of JPG files and compressed numpy arrays of `.npz` files)
- `front/src/assets/` (JSON files that specify genomic ranges)
- `front/public/assets/` (CSV files that contain external analysis results)

These datasets are shared upon request.
