# CS 598 Deep Learning For Health Care
## Project Overview 
This repository contains the code and documentation of our CS 598 final project. For our project, we attempt to recreate the results of the paper "Natural language processing for cognitive therapy: Extracting schemas from thought records" by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman.

The paper is available for download on the [PLOS ONE website](https://app.dimensions.ai/details/publication/pub.1141955424]) 

## Repository Structure
├── data 				- a subset of the paper's dataset (see below for access instructions) 
├── papers 				- papers under consideration for this project 
├── reference_material	- guidance documents 
├── src 				- all code used in this project 
│   └── cog_therapy.py 	
├── submissions 		- final documents submitted to the class 
├── requirements.txt  	- pip requirements file 


## Data Access 
The authors full dataset and code is available for download on the [4TU.ResearchData repository](https://data.4tu.nl/articles/dataset/Dataset_and_Analyses_for_Extracting_Schemas_from_Thought_Records_using_Natural_Language_Processing/16685347).

Data can be downloaded directly from this website as a zip file. To verify using the researchers' original data. Download the zip file. Extract the file to a known location, then retreive the folder ```Data/DatasetsForH1/``` and place it into this repository's ```data``` folder 

## Project Setup 
1. Download and install [Python 3.10.4](https://www.python.org/downloads/)
1. Create a new [virtual environment](https://docs.python.org/3/library/venv.html) and activate it
1. Navigate a python terminal to the root directory of this repository and run ```pip install requirements.txt```\
1. (Optional) we recommend downloading the [spyder IDE](https://docs.spyder-ide.org/current/installation.html) for streamlined debugging 

## Project Execution
Using a python terminal, run ```python cog_therapy.py```

## Project Dependencies 
This project relies on the following python modules (pip format): 
