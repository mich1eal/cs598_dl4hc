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
import numpy as np
import torchtext
import torch 
import torch.nn as nn
# To help perform hyperparameter grid search
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
# To compute model goodness-of-fit
from scipy.stats import spearmanr


# Set seed. Copied from HW3 RNN notebook.
seed = 24
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#os.environ["PYTHONHASHSEED"] = str(seed)


DATA_DIR = '../data/DatasetsForH1'
DATASETS = ['H1_test', 'H1_train', 'H1_validate'] 
SCHEMAS = ["Attach","Comp","Global","Health","Control","MetaCog","Others","Hopeless","OthViews"]

# used for creating text dictionaries 
PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>' #not used in nominal model 

test_fraction = .15
val_fraction = .1275
train_fraction = 1 - test_fraction - val_fraction

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

# we combine all data into one dataframe for convenient preprocessing 
label_frame = pd.concat(label_frames, axis=0)
text_frame = pd.concat(text_frames, axis=0)

in_frame = pd.concat([text_frame, label_frame], axis=1)


###### Preprocess data
# we will tokenize using pytorch utility function
tokenize = torchtext.data.get_tokenizer('basic_english', language='en')

# lists of tokens are re-added to our dataframe 
in_frame['tokens'] = [tokenize(sentence) for sentence in in_frame['Utterance']]

split_train = int(len(in_frame) * train_fraction)
split_val = split_train + int(len(in_frame) * val_fraction)

train_frame = in_frame.iloc[:split_train]
val_frame = in_frame.iloc[split_train:split_val]
test_frame = in_frame.iloc[split_val:]


###### Prepare dataloader
# we define a custom text dataset for use with all models 
class CognitiveDataset(torch.utils.data.Dataset):
    def __init__(self, in_frame, split, max_len, threshold=1, idx2word=None, word2idx=None):
        '''
        in_frame - a dataframe with columns defined above 
        split - one of {'train', 'val', 'test'}
        max_len - number of tokens to crop/pad sentences to
        threshold - the minimum number of times a token must be seen to be added to the dictionary
        idx2word, word2idx - generated here, used for mapping tokens and indeces 
        '''
        
        self.utterances = in_frame['tokens'].to_list()
        self.labels = in_frame[SCHEMAS].to_numpy()
        self.threshold = threshold
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.max_len = max_len

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    def build_dictionary(self): 
        '''
        Build dictionaries idx2word and word2idx. This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. 
        '''
        assert self.split == 'train'
        
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}

        # Count the frequencies of all words in the training data 
        token_freq = {}
        idx = 3

        for utterance in self.utterances:
            for token in utterance:

                #keep track of word frequency 
                if token in token_freq:
                    token_freq[token] += 1
                else:
                    token_freq[token] = 1

                # once we have seen word enough times, add it to index
                if token_freq[token] == self.threshold:
                    self.idx2word[idx] = token
                    self.word2idx[token] = idx
                    idx += 1
    
    def convert_text(self):
        '''
        Convert each utterance to a list of indices
        '''
        
        # Get unknown and end of sentence indeces for efficiency 
        unk_token = self.word2idx[UNK]
        eos_token = self.word2idx[END]
        
        for utterance in self.utterances:
            idx_list = []        
      
            for token in utterance:
                #get index of token, if not in list use uknown token
                idx_list.append(self.word2idx.get(token, unk_token))
          
            #always add EOS tag
            idx_list.append(eos_token)
            self.textual_ids.append(idx_list)
      

    def get_text(self, idx):
        '''
        Return the utterance at idx as a long tensor of integers 
        corresponding to the words in the utterance.
        Adds padding as required
        '''
        
        indices = self.textual_ids[idx]
        idx_len = len(indices)

        if idx_len > self.max_len:
            indices = indices[:self.max_len]
            #too long, trim

        elif idx_len < self.max_len:
            #too short, add padding. Assumes padding idx is 0
            pad_list = [self.word2idx[PAD]] * (self.max_len - idx_len)
    
            indices = indices + pad_list

        #otherwise length is correct, no action
        return torch.LongTensor(indices)
    
    def get_label(self, idx):
        '''
        Return labels as a long vector 
        '''
        return torch.LongTensor(self.labels[idx])

    def __len__(self):
        '''
        Return the number of utterances in the dataset
        '''
        return len(self.examples)
    
    def __getitem__(self, idx):
        '''
        Return the utterance, and label of the review specified by idx.
        '''
        return self.get_text(idx), self.get_label(idx)


train_set = CognitiveDataset(train_frame, 'train', threshold=1, max_len=25)

val_set = CognitiveDataset(val_frame, 'val', threshold=1, max_len=25, 
                           idx2word=train_set.idx2word, 
                           word2idx=train_set.word2idx)
test_set = CognitiveDataset(test_frame, 'test', threshold=1, max_len=25,  
                           idx2word=train_set.idx2word, 
                           word2idx=train_set.word2idx)

###### Evaluation routines

def spearman_r(X, Y):
    '''
    Computes Spearman rank-order correlation between
    model predictions and ground truth.
    Code taken from researchers' original Jupyter notebook.
    
    Inputs:
        X: matrix of predictions, with one column per schema.
        Y: matrix of ground-truth labels, with one column per schema.
    Output:
        Numpy array of correlation coefficients, one per schema
    '''
    
    # Initialize array of corr coefficients
    rho_array = np.zeros(X.shape[1])

    # Compute coefficient over each schema and save
    for schema in range(len(SCHEMAS)):
        rho, p_val = spearmanr(X[:, schema], Y[:, schema])
        rho_array[schema] = rho

    return rho_array

###### Baseline models



###### Paper primary model

### Multi-label RNN

class MultiLabelRNN(nn.Module):
    
    '''
    PyTorch implementation of the researchers' multi-label RNN.
    This is a bidirectional LSTM with a dropout layer and a dense
    output layer, with sigmoid activation.
    
    For now, use torch nn.Embedding instead of GLoVE embeddings.
    Will need to replace this with GLoVE embeddings.
    '''
    
    def __init__(self, vocab_size, embed_size=100, hidden_size=100, dropout=0.5, num_labels=len(SCHEMAS)):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear((2 * hidden_size), num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hidden_state_n, cell_state_n) = self.lstm(embeds)
        do_out = self.dropout(lstm_out)
        label_probs = self.sigmoid(self.fc(do_out))
        
        return label_probs


### TO DO

# 0) Create dataloaders for train, validation and test sets
# 1) Proof of concept -- train multi-label RNN
# 2) Tune hyperparameters using grid search
#    Possible ways to do this: https://medium.com/pytorch/accelerate-your-hyperparameter-optimization-with-pytorchs-ecosystem-tools-bc17001b9a49
#        https://discuss.pytorch.org/t/what-is-the-best-way-to-perform-hyper-parameter-search-in-pytorch/19943/3
#        https://skorch.readthedocs.io/en/stable/user/quickstart.html
#        Use skorch package to wrap PyTorch in SciKit-Learn to take advantage of sklearn functionality
# 3) Get model with best mean absolute error
# 4) Compute Spearman rank-order correlation between model predictions and ground truth for each schema
# 5) Train 30 models with same parameters as best model. Report the mean correlations.

# Routine to perform grid search over RNN parameters.
# Use skorch package to implement grid search.

def RNNGridSearch():
    
    # Specs for grid search.
    # NOTE: this does not include learning rate, optimizer type, or loss function types.
    params = {'module__hidden_size': [50,100],
              'module__dropout': [0.1, 0.5],
              'batch_size': [32, 64],
              'epochs': [100]}



###### Ablation study 