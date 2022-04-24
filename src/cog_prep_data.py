""" 
cog_read.py

Routines to read in, tokenize and embed the selected cognitive-therapy dataset.
Data can be either the authors' original prepreocessed set, or our own
preprocessed set.

Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/24/2022

Paper: "Natural language processing for cognitive therapy: Extracting schemas from thought records"
by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman
DOI: https://doi.org/10.1371/journal.pone.0257832
Data repo: https://github.com/mich1eal/cs598_dl4hc
For dependencies, and data acquisition instructions, please see this repository's readme
"""

import pandas as pd
# Get tokenizers for each potential language model
from torchtext.data import get_tokenizer
from transformers import BertTokenizer

def read_data(data_dir='../data/DatasetsForH1', datasets=['H1_test', 'H1_train', 'H1_validate']):
    '''
    Read data and label files for cognitive therapy paper.
    Assumes each filename ends with either '_data.csv' or '_labels.csv'.
    Input:
        data_dir: location of files
        datasets: list of strings to match files by. Each string matches a data file plus a label file.
    Output: single combined file of data and labels
    '''
    
    label_frames = []
    for dataset in datasets: 
        file_path = '{}/{}_labels.csv'.format(data_dir, dataset)
        label_frames.append(pd.read_csv(file_path, sep=';', header=0))

    # read in all texts 
    text_frames = []
    for dataset in datasets: 
        file_path = '{}/{}_texts.csv'.format(data_dir, dataset)
        text_frames.append(pd.read_csv(file_path, sep=';', header=0))

    # we combine all data into one dataframe for convenient preprocessing 
    label_frame = pd.concat(label_frames, axis=0)
    text_frame = pd.concat(text_frames, axis=0)

    return pd.concat([text_frame, label_frame], axis=1)

def return_tokenizer(lang_model=None):
    '''
    Return tokenizer for the specified language model.
    If BERT, return BERT's base-uncased tokenizer.
    Otherwise return torchtext's basic-english tokenizer.
    
    Input: lang_model: 'BERT', 'GLoVe', None
    Output: tokenizer
    '''
    if lang_model == 'BERT':
        return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    else:  # Else return torchtext tokenizer
        return get_tokenizer('basic_english', language='en')

  
def tokenize_data(dataframe,
                  lang_model=None,
                  max_length=25,
                  pad=True,
                  add_special_tokens=False,
                  special_tokens=None,
                  default_token=None):
    '''
    Given a dataframe that includes utterances, add a 'tokens' column
    that contains a tokenized list of utterances.
    If BERT is the language model, also return a 'masks' column containing a list of masks.

    Inputs:
        dataframe: the source dataframe. Must contain an 'Utterance' column.
        lang_model: 'GLoVe', 'BERT', or None.
        max_length: maximum length to pad utterance to.
        pad: whether to pad each utterance to max_length.
        add_special_tokens: True to incorporate special tokens into utterance.
        special_tokens: list of special tokens, if required.
            list not required for BERT, which will always use [CLS] and [SEP].
        default_token: default for tokens outside vocabulary. Not required for BERT.
    Outputs: the input dataframe with new 'tokens' column, plus 'masks' column if language model is BERT.
    '''
    tokenizer = return_tokenizer(lang_model)
    tokens = []

    # Tokenize for BERT.
    # Code borrowed from http://mccormickml.com/2019/07/22/BERT-fine-tuning/#51-data-preparation
    if lang_model == 'BERT':
        masks = []
        
        for utterance in dataframe['Utterance']:

            encoded_dict = tokenizer.encode_plus(
                        utterance,                                # Sentence to encode.
                        add_special_tokens = add_special_tokens,  # Add '[CLS]' and '[SEP]'
                        max_length = max_length,                  # Pad & truncate all sentences.
                        pad_to_max_length = pad,
                        return_attention_mask = True              # Construct attn. masks.
                        #return_tensors = 'pt',                   # Uncomment to return PyTorch tensors instead of lists                   )
            )

            tokens.append(encoded_dict['input_ids'])
            masks.append(encoded_dict['attention_mask'])
            
        dataframe['tokens'] = tokens
        dataframe['masks'] = masks
        
    # For other language model or no language model, use torchtext tokenizer.
    # NOT IMPLEMENTED, TO DISCUSS: whether to do remaining tokenization tasks:
    #truncation, padding, adding special characters, or whether to do them in CognitiveDataset.
    # This first requires building a vocabulary.
    # 
    else:
        dataframe['tokens'] = [tokenizer(utterance) for utterance in dataframe['Utterance']]
    
    return dataframe


def embed_data():
    '''
    Given tokenized input and a vocabulary, return embeddings.
    NOT IMPLEMENTED YET.

    This is only needed for GLoVe or index-based embeddings.
    With BERT models, don't need to deal directly with embeddings.

    '''