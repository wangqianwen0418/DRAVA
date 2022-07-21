# Backend

The backend is developed and tested with `python@3.7.9`

First, download or clone our github repo at https://github.com/wangqianwen0418/DRAVA.

Second, go to the repo folder and install all dependent packages.  
To manage dependencies more effectively, you can create and use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) before installing all packages:

```sh
cd server
conda create -n drava python==3.7.9
conda deactivate && conda activate drava
pip install -r requirements.txt
```

Then, start the flask server:

```sh
cd flask_server
python app.py
```

The pre-trained models are stored at `Drava/server/flask_server/saved_models`.
You can download our pretrained models at [here](https://drive.google.com/drive/folders/11K-v8Fn4PbbRqCcrRpLSsnvxBaOPH1db?usp=sharing).

The datasets are stored at `server/data`.
You can download the example datsets at [here](https://drive.google.com/drive/folders/16kbJq_46-4Busrz_87vGFyKAsy15oIU3?usp=sharing). 