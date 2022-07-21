# Quick Start

The easiest way to run Drava on your computer is to download the [zip file from Google Drive](https://drive.google.com/file/d/10I4cf2TMXX76f33UEz6GTI9te9UQKcHR/view?usp=sharing) and run it using Python.

1. Download and unzip the file to your local computer
1. Start a virtual environment and install dependency packages
   ```bash
   conda create -n drava python==3.7.9
   conda deactivate && conda activate drava
   pip install -r requirements.txt
   ```
1. Run the demo
   ```bash
   cd flask-server
   python run app.py
   ```
    You can open the demo at `http://localhost:8080` in your web browser.

