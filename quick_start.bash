#!/bin/bash
echo "generate a quick_start zip file from the current code"

# build frontend and move it to the flask server 
cd front
npm run build
cd ../
rm -r server/flask_server/build/
mv front/build/ server/flask_server

# download the minimized datasets if not exist
if [ ! -d "data/" ] 
then 
    echo "please download the data folder from 'https://drive.google.com/drive/folders/16kbJq_46-4Busrz_87vGFyKAsy15oIU3?usp=sharing', unzip it in the root folder, and rename it as data/"
else 
    # generate quick_start.zip
    zip -r drava_flask.zip server/requirements.txt server/dataloaders.py server/experiment.py server/utils.py server/flask_server/* server/models/__init__.py server/models/base.py server/models/beta*.py server/models/types_.py server/ConceptAdaptor.py data/* -x *.DS_Store */__pycache__/* @
fi

