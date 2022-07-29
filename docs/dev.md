# Run Drava in Development Mode
First, download or clone our github repo at https://github.com/wangqianwen0418/DRAVA.

## Frontend

The front-end visual interface is developed and tested using node@v16.10.0 at Chrome web browser.


- Go to the repo front-end folder and install all dependent packages:

```
cd front
npm install
```

- Then, launch the Drava react application on the browser:

```
npm start
```

## Backend

The backend is developed and tested with `python@3.7.9`


- Go to the repo folder and install all dependent packages.  
To manage dependencies more effectively, you can create and use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) before installing all packages:

```
cd server
conda create -n drava python==3.7.9
conda deactivate && conda activate drava
pip install -r requirements.txt
```

- Start the flask server:

```
cd flask_server
python app.py
```

- Example Data  
  
    The pre-trained models are stored at `Drava/server/flask_server/saved_models`.
    You can download our pretrained models at [here](https://drive.google.com/drive/folders/11K-v8Fn4PbbRqCcrRpLSsnvxBaOPH1db?usp=sharing).

    The datasets are stored at `server/data`.
    You can download the example datsets at [here](https://drive.google.com/drive/folders/16kbJq_46-4Busrz_87vGFyKAsy15oIU3?usp=sharing). 