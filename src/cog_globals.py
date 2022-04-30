"""
Return global constants/variables for analysis of cognitive-therapy data.

Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Paper: "Natural language processing for cognitive therapy: Extracting schemas from thought records"
by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman
DOI: https://doi.org/10.1371/journal.pone.0257832
Data repo: https://github.com/mich1eal/cs598_dl4hc
For dependencies, and data acquisition instructions, please see this repository's readme
"""

# To set random seed
seed = 24

# Column names for schema ratings
SCHEMAS = ["Attach","Comp","Global","Health","Control","MetaCog","Others","Hopeless","OthViews"]

# List of values for schema correspondence ratings
RATING_VALS = [0, 1, 2, 3]

#Locations
DATA_DIR = '../data/DatasetsForH1'
DEFAULT_DATASETS = ['H1_test', 'H1_train', 'H1_validate']
CUSTOM_DATASETS = ['H1_custom']

# used for creating torchtext dictionaries 
UNK = '<UNK>' #not used in nominal model 
END = '<END>'
PAD = '<PAD>'
SPECIAL_TOKENS = [UNK, END, PAD]

# Used for creating BERT-tokenized utterances.
# The encode_plus() method uses these automatically; they are here for reference.
CLS = '[CLS]'  # For BERT-based classification models
SEP = '[SEP]'  # To separate sentence pairs in textual input

# Train / validation / test split ratios
test_fraction = .15
val_fraction = .1275
train_fraction = 1 - test_fraction - val_fraction

# Settings for utterances, vocabulary and embeddings

max_utt_length = 25
max_vocab_size = 2000

# Embedding dimensions for GLoVe
glove_embed_dim = 100