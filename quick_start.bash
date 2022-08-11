#!/bin/bash

# build frontend and move it to the flask server 
cd front
npm run build
cd ../
rm -r server/flask_server/build/
mv front/build/ server/flask_server

# download the minimized datasets if not exist
if [ ! -d "data/" ] 
then echo "please download the data folder from 'https://drive.google.com/drive/folders/16kbJq_46-4Busrz_87vGFyKAsy15oIU3?usp=sharing', unzip it in the root folder, and rename it as data/"
fi

cd server
# generate quick_start.zip
zip -r ../drava_flask.zip requirements.txt dataloaders.py experiment.py utils.py flask_server/* models/__init__.py models/base.py models/beta*.py ../data/*
