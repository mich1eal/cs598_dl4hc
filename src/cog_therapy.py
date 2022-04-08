""" 
Michael Miller and Kurt Touhy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Paper: "Natural language processing for cognitive therapy: Extracting schemas from thought records"
by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman
DOI: https://doi.org/10.1371/journal.pone.0257832
Data repo: https://github.com/mich1eal/cs598_dl4hc
For dependencies, and data acquisition instructions, please see this repository's readme
"""

import pandas as pd


DATA_DIR = '../data/DatasetsForH1'
DATASETS = ['H1_test', 'H1_train', 'H1_validate'] 

###### Load data
# read in all labels
label_frames = []
for dataset in DATASETS: 
    file_path = '{}/{}_labels.csv'.format(DATA_DIR, dataset)
    label_frames.append(pd.read_csv(file_path, sep=';', header=0))

# read in all texts 
text_frames = []
for dataset in DATASETS: 
    file_path = '{}/{}_texts.csv'.format(DATA_DIR, dataset)
    text_frames.append(pd.read_csv(file_path, sep=';', header=0))

# we combine split data into one dataframe for convenience preprocessing 
label_frame = pd.concat(label_frames, axis=0)
text_frame = pd.concat(text_frames, axis=0)

###### Baseline models


###### Paper primary model


###### Ablation study 