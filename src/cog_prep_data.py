""" 
Routines to read in, tokenize, split and embed the selected cognitive-therapy dataset.
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
import numpy as np
# Get tokenizers for each potential language model
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchtext
import cog_globals as GLOB
from tensorflow.keras.preprocessing.text import Tokenizer
from autocorrect import Speller
from itertools import compress 
spell = Speller(lang='en')


# Routine to generate dataloader
def create_dataloader(dataset, batch_size=32, shuffle=False):
    '''
    Return a dataloader for the specified dataset and batch size.
    Very similar to most CS 598 DLH homework problems.
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class TokenDataset(Dataset):
    def __init__(self, in_frame, schemas=GLOB.SCHEMAS, max_len=GLOB.max_utt_length, vocab_size=GLOB.max_vocab_size, vocab=None, embeddings=None):
        '''
        Torch Dataset that embeds at the token level
        in_frame - a dataframe with columns defined above 
        schemas - list of schemas to generate labels for. If a single schema, use one-hot encoding of its rating values.
        max_len - number of tokens to crop/pad sentences to
        vocab_size - will reduce the number of words to this value
        vocab - either a torchtext.vocab.Vocab object, or None to build one
        embeddings - a torchtext.Vocab.Vector object or None for indices
        '''
        
        self.utterances = in_frame['tokens'].to_list()
        
        # For single schema, generate one-hot encoding of rating values for labels
        if len(schemas) == 1:
            schema_rating_vals = torch.LongTensor(np.array(in_frame[schemas])).squeeze(dim=1)
            self.labels = F.one_hot(schema_rating_vals, num_classes=len(GLOB.RATING_VALS)).numpy()
        # Otherwise use original ratings as labels
        else:
            self.labels = in_frame[schemas].to_numpy()
            
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Dictionaries
        self.vocab = vocab
        if vocab is None:
            self.build_vocab(embeddings)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    def build_vocab(self, embeddings): 
        '''
        Build torch dictionary. This is only called when split='train', as the 
        vocab is passed in to the __init__(...) function otherwise. 
        '''        
        # Count the frequencies of all words in the training data 
        token_list = []
        for utterance in self.utterances:
            token_list.extend(utterance)
        
        #note that torchtext.vocab.vocab works for chars. To work with strings, 
        # add double nested list 
        token_list_list = [[token] for token in token_list]
            
        self.vocab = torchtext.vocab.build_vocab_from_iterator(token_list_list, 
                           max_tokens=self.vocab_size,
                           specials=[GLOB.UNK, GLOB.END, GLOB.PAD])
        self.vocab.set_default_index(self.vocab[GLOB.UNK])
        
        if embeddings is not None:
            #Get the relevent rows in GLoVE, and match their indices with
            #the indices in our vocab https://github.com/pytorch/text/issues/1350
            self.embed_vec = embeddings.get_vecs_by_tokens(self.vocab.get_itos())

    def convert_text(self):
        '''
        Convert each utterance to a list of indices
        '''
        for utterance in self.utterances:
            idx_list = [self.vocab[token] for token in utterance]
            self.textual_ids.append(idx_list)

    def get_text(self, idx):
        '''
        Return the utterance per the type of ebedding specified
        Adds padding as required
        '''
        indices = self.textual_ids[idx]

            #no embedding required, return just indices, but pad to correct length 
            
        indices.append(self.vocab[GLOB.END])
        idx_len = len(indices)
        
        if idx_len > self.max_len:
            #too long, trim
            indices = indices[:self.max_len]

        elif idx_len < self.max_len:
            #too short, add padding
            indices += [self.vocab[GLOB.PAD]] * (self.max_len - idx_len)
                    
        return torch.LongTensor(indices)
        
    def get_label(self, idx):
        '''
        Return labels as a long vector 
        '''
        return torch.FloatTensor(self.labels[idx])

    def __len__(self):
        '''
        Return the number of utterances in the dataset
        '''
        return len(self.utterances)
    
    def __getitem__(self, idx):
        '''
        Return the utterance, and label of the review specified by idx.
        '''
        return self.get_text(idx), self.get_label(idx)

class UtteranceDataset(Dataset):
    def __init__(self, in_frame, schemas=GLOB.SCHEMAS, max_len=GLOB.max_utt_length, vocab_size=GLOB.max_vocab_size, vocab=None, tfidf_tokenizer=None, embeddings=None):
        '''
        Torch Dataset that embeds at the utterance level
        in_frame - a dataframe with columns defined above 
        schemas - list of schemas to generate labels for. If a single schema, use one-hot encoding of its rating values.
        max_len - number of tokens to crop/pad sentences to
        vocab_size - will reduce the number of words to this value
        vocab - either a torchtext.vocab.Vocab object, or None to build one
        tfidf_tokenizer - either a keras Tokenizer, or None to build one
        embeddings - a torchtext.Vocab.Vector object or None for indices
        '''
        
        #must have both or neither 
        assert (vocab and tfidf_tokenizer) or (not vocab and not tfidf_tokenizer)
        self.utterances = in_frame['tokens'].to_list()
        # For single schema, generate one-hot encoding of rating values for labels
        if len(schemas) == 1:
            schema_rating_vals = torch.LongTensor(np.array(in_frame[schemas]))
            self.labels = F.one_hot(schema_rating_vals, num_classes=len(GLOB.RATING_VALS)).numpy()
        # Otherwise use original ratings as labels
        else:
            self.labels = in_frame[schemas].to_numpy()
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Dictionaries
        self.vocab = vocab
        if vocab is None:
            self.build_vocab(embeddings)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()
        
        # Set embed vec
        if embeddings is not None:
            #Get the relevent rows in GLoVE, and match their indices with
            #the indices in our vocab https://github.com/pytorch/text/issues/1350
            self.embed_vec = embeddings.get_vecs_by_tokens(self.vocab.get_itos())
        
        
        #tfidf embeddings
        self.tfidf_tokenizer = tfidf_tokenizer
        if tfidf_tokenizer is None: 
            # Get TFIDF tokenizer if it wasn't provided
            tfidf_tokenizer = Tokenizer(oov_token = GLOB.UNK,
                                  num_words=vocab_size)
            tfidf_tokenizer.fit_on_sequences(self.textual_ids)
            self.tfidf_tokenizer = tfidf_tokenizer
        
        #get tfidf embeddings 
        tfidf = torch.Tensor(self.tfidf_tokenizer.sequences_to_matrix(self.textual_ids, mode='tfidf'))
        
        # Normalize embeddings
        tfidf = F.normalize(tfidf, dim=1)
        
        # Make embeddings sum to 1
        tfidf = tfidf / tfidf.sum(dim=1).unsqueeze(-1)
        
        # dot product with saved embeddings  
        utterance_embeddings = torch.mm(tfidf, self.embed_vec)
        
        # delete edge case where all 0s
        drop_idx = utterance_embeddings.sum(dim=1) == 0
        self.utterance_embeddings = utterance_embeddings[~drop_idx, :] 
        self.labels = self.labels[~drop_idx, :] 
        
        
    def build_vocab(self, embeddings): 
        '''
        Build torch dictionary. This is only called when split='train', as the 
        vocab is passed in to the __init__(...) function otherwise. 
        '''        
        # Count the frequencies of all words in the training data 
        token_list = []
        for utterance in self.utterances:
            token_list.extend(utterance)
        
        #note that torchtext.vocab.vocab works for chars. To work with strings, 
        # add double nested list 
        token_list_list = [[token] for token in token_list]
            
        self.vocab = torchtext.vocab.build_vocab_from_iterator(token_list_list, 
                           max_tokens=self.vocab_size,
                           specials=[GLOB.UNK, GLOB.END, GLOB.PAD])
        self.vocab.set_default_index(self.vocab[GLOB.UNK])
        
            
    def convert_text(self):
        '''
        Convert each utterance to a list of indices
        '''
        for utterance in self.utterances:
            idx_list = [self.vocab[token] for token in utterance]
            self.textual_ids.append(idx_list)

    def __len__(self):
        '''
        Return the number of utterances in the dataset
        '''
        return self.utterance_embeddings.shape[0]
    
    def __getitem__(self, idx):
        '''
        Return the utterance, and label of the review specified by idx.
        '''
        return self.utterance_embeddings[idx, :], torch.FloatTensor(self.labels[idx])



def read_data(process_mode=None):
    '''
    Read data and label files for cognitive therapy paper.
    Input:
        process_mode:
            - None to use authors processing
            - 'utterance' to keep each utterance as its own line
            - 'scenario' to concatenate utterances for one participant scenario
        
    Output: single combined file of data and labels
    '''
    if not process_mode:
        # Use the authors original preprocessed data
        label_frames = []
        for dataset in GLOB.DEFAULT_DATASETS: 
            file_path = '{}/{}_labels.csv'.format(GLOB.DATA_DIR, dataset)
            frame = pd.read_csv(file_path, sep=';', header=0)
            label_frames.append(frame)
    
        # read in all texts 
        text_frames = []
        for dataset in GLOB.DEFAULT_DATASETS: 
            file_path = '{}/{}_texts.csv'.format(GLOB.DATA_DIR, dataset)
            frame = pd.read_csv(file_path, sep=';', header=0)
            text_frames.append(frame)
    
        # we combine all data into one dataframe for convenient preprocessing 
        label_frame = pd.concat(label_frames, axis=0)
        text_frame = pd.concat(text_frames, axis=0)
    
        return pd.concat([text_frame, label_frame], axis=1)
    else: 
        assert process_mode in ['utterance', 'scenario']
        
        # use custom preprocessed data (see cog_preprocess.py)
        file_path = '{}/{}'.format(GLOB.DATA_DIR, GLOB.CUSTOM_PREPROCESS)
        in_frame = pd.read_csv(file_path)
        
        if process_mode == 'scenario':
            #group by participant and and scenario
            grouped = in_frame.groupby(['Participant.ID', 'Scenario'], as_index = False)
            
            group_map = {col : 'mean' for col in GLOB.SCHEMAS}
            group_map['Utterance'] = ' '.join
            
            in_frame = grouped.agg(group_map)
        
        #only keep rows that we need
        out_frame = in_frame[['Utterance'] + GLOB.SCHEMAS]
        
        #shuffle the frame
        out_frame = out_frame.sample(frac=1).reset_index(drop=True)
        return out_frame
        
def tokenize_bert(dataframe,
                  max_length=25,
                  pad=True,
                  add_special_tokens=False,
                  special_tokens=None,
                  default_token=None):
    '''
    Given a dataframe that includes utterances, add a 'tokens' column
    that contains a tokenized list of utterances.
    Also return a 'masks' column containing a list of masks.

    Inputs:
        dataframe: the source dataframe. Must contain an 'Utterance' column.
        max_length: maximum length to pad utterance to.
        pad: whether to pad each utterance to max_length.
        add_special_tokens: True to incorporate special tokens into utterance.
        special_tokens: list of special tokens, if required.
            list not required for BERT, which will always use [CLS] and [SEP].
        default_token: default for tokens outside vocabulary. Not required for BERT.
    Outputs: the input dataframe with new 'tokens' column, plus 'masks' column.
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokens = []
    masks = []

    # Tokenize for BERT.
    # Code borrowed from http://mccormickml.com/2019/07/22/BERT-fine-tuning/#51-data-preparation
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
    return dataframe

def tokenize(dataframe):
    '''
    Given a dataframe that includes utterances, add a 'tokens' column
    that contains a tokenized list of utterances.
    
    Inputs:
        dataframe: the source dataframe. Must contain an 'Utterance' column.
    Outputs: the input dataframe with new 'tokens' column.
    '''
    # we will tokenize using pytorch utility function
    tokenizer = torchtext.data.get_tokenizer('basic_english', language='en')
    # lists of tokens are re-added to our dataframe 
    dataframe['tokens'] = [tokenizer(sentence) for sentence in dataframe['Utterance']]
    
    return dataframe

def split_data(dataframe, train_fraction=GLOB.train_fraction, val_fraction=GLOB.val_fraction, test_fraction=GLOB.test_fraction):
    '''
    Given a dataframe, return training, validation and test splits
    with the specified proportions.
    '''
    
    split_train = int(len(dataframe) * train_fraction)
    split_val = split_train + int(len(dataframe) * val_fraction)

    train_frame = dataframe.iloc[:split_train]
    val_frame = dataframe.iloc[split_train:split_val]
    test_frame = dataframe.iloc[split_val:]
    
    return (train_frame, val_frame, test_frame)
