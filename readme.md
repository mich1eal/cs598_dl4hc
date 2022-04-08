# CS 598 Final Project
## Overview 
This repository contains the code and documentation of our CS 598 final project. For this project, we recreate the results of the paper "Natural language processing for cognitive therapy: Extracting schemas from thought records" by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman.

The paper is available for download on the [PLOS ONE](https://app.dimensions.ai/details/publication/pub.1141955424]) website. 

## Repository Structure
├── data - a subset of the paper's dataset (see below for access instructions)  
├── papers - papers under consideration for this project  
├── reference_material - guidance documents  
├── src - all code used in this project  
│   └── cog_therapy.py  
├── submissions - final documents submitted to the class  
├── requirements.txt - pip requirements file  

## Data Access 
The authors' full dataset and code is available for download on the [4TU.ResearchData](https://data.4tu.nl/articles/dataset/Dataset_and_Analyses_for_Extracting_Schemas_from_Thought_Records_using_Natural_Language_Processing/16685347) repository.

Data can be downloaded directly from this website as a zip file. To verify using the researchers' original data: 
1. Download the zip file
1. Extract the file to a known location
1. Retrieve the folder ```Data/DatasetsForH1/``` and use it to overwrite this repository's ```data``` folder 

## Project Setup 
1. Download and install [Python 3.10.4](https://www.python.org/downloads/)
1. Create a new [virtual environment](https://docs.python.org/3/library/venv.html) and activate it
1. Navigate a python terminal to the root directory of this repository and run ```pip install -r requirements.txt```
1. (Optional) Download [Spyder IDE](https://docs.spyder-ide.org/current/installation.html) and [configure it to run with the venv](https://medium.com/analytics-vidhya/5-steps-setup-python-virtual-environment-in-spyder-ide-da151bafa337) 

## Project Execution
Using a python terminal, run ```python cog_therapy.py```

## Project Dependencies 
This project relies on the following python modules (pip format): 
backcall==0.2.0
cloudpickle==2.0.0
colorama==0.4.4
debugpy==1.6.0
decorator==5.1.1
entrypoints==0.4
gensim==4.1.2
ipykernel==6.12.1
ipython==7.32.0
jedi==0.18.1
joblib==1.1.0
jupyter-client==7.2.2
jupyter-core==4.9.2
matplotlib-inline==0.1.3
nest-asyncio==1.5.5
numpy==1.22.3
packaging==21.3
pandas==1.4.2
parso==0.8.3
pickleshare==0.7.5
prompt-toolkit==3.0.29
psutil==5.9.0
Pygments==2.11.2
pyparsing==3.0.7
python-dateutil==2.8.2
pytz==2022.1
pywin32==303
pyzmq==22.3.0
scikit-learn==1.0.2
scipy==1.8.0
six==1.16.0
smart-open==5.2.1
spyder-kernels==2.3.0
threadpoolctl==3.1.0
torch==1.11.0
tornado==6.1
traitlets==5.1.1
typing_extensions==4.1.1
wcwidth==0.2.5