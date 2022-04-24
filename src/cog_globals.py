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

# Column names for schema ratings
SCHEMAS = ["Attach","Comp","Global","Health","Control","MetaCog","Others","Hopeless","OthViews"]

#Locations
DATA_DIR = '../data/DatasetsForH1'
DEFAULT_DATASETS = ['H1_test', 'H1_train', 'H1_validate']
CUSTOM_DATASETS = ['H1_custom']

# used for creating torchtext dictionaries 
UNK = '<UNK>' #not used in nominal model 
END = '<END>'
PAD = '<PAD>'

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

# Return globals appropriate to the specified language model.

def return_globals(use_custom_dataset=False, lang_model=None):
    '''
    Given a language model name, return a dict of settings appropriate for that model.
    
    '''
    
    globals_dict = {
        'SCHEMAS': SCHEMAS,
        'DATA_DIR': DATA_DIR,
        'test_fraction': test_fraction,
        'val_fraction': val_fraction,
        'train_fraction': train_fraction
        }
        
    globals_dict['DATASETS'] = CUSTOM_DATASETS if use_custom_dataset else DEFAULT_DATASETS
    
    if lang_model is None or lang_model != 'BERT':
        globals_dict['UNK'] = UNK
        globals_dict['END'] = END
        globals_dict['PAD'] = PAD
        globals_dict['special_tokens'] = [UNK, END, PAD]
        globals_dict['max_utt_length'] = max_utt_length
        globals_dict['max_vocab_size'] = max_vocab_size

    return globals_dict