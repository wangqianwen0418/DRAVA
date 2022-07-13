# Backend

The backend is developed and tested with `python@3.7.9`

First, download or clone our github repo at https://github.com/wangqianwen0418/DRAVA.

Second, go to the repo folder and install all dependent packages:

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
You can download our pretrained models at {TODO:}
