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

from collections import Counter
import pandas as pd
import numpy as np
import torchtext
from torchtext.vocab import GloVe
from torchtext.vocab import Vocab
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
# To help perform hyperparameter grid search
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
# To compute model score on validation set
from sklearn.metrics import mean_absolute_error
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
    def __init__(self, in_frame, max_len, vocab_size=2000, vocab=None, embeddings=None, embed_mode=None):
        '''
        in_frame - a dataframe with columns defined above 
        max_len - number of tokens to crop/pad sentences to
        vocab_size - will reduce the number of words to this value
        vocab - either a torchtext.vocab.Vocab object, or None to build one
        embeddings - if embed_mode is 'token' or 'utterance', used to create word embeddings
        embed_mode: 
            'token' - embed each token, return as tensor of shape(max_len, emb_dim)
            'utterance' - embed each utterance, return as tensor of shape (max_len)
            None - no embedding. return utterance as list of ints of length max_len
        '''
        
        self.utterances = in_frame['tokens'].to_list()
        self.labels = in_frame[SCHEMAS].to_numpy()
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Dictionaries
        self.vocab = vocab
        if vocab is None:
            self.build_vocab(embeddings)
        
        assert embed_mode in {'token', 'utterance', None}
        self.embed_mode = embed_mode 
        
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
            
        token_counter = Counter(token_list)
            
        self.vocab = Vocab(token_counter, 
                           max_size=self.vocab_size,
                           specials=[PAD, END, UNK],
                           vectors=self.embeddings)

    def convert_text(self):
        '''
        Convert each utterance to a list of indices
        '''
        for utterance in self.utterances:
            idx_list = [self.vocab.stoi[token] for token in utterance]
      
            idx_list.append(self.vocab.stoi[END])
            
            self.textual_ids.append(idx_list)
                       

    def get_text(self, idx):
        '''
        Return the utterance per the type of ebedding specified
        Adds padding as required
        '''
        
        indices = self.textual_ids[idx]
        idx_len = len(indices)

        if idx_len > self.max_len:
            #too long, trim
            indices = indices[:self.max_len]

        elif idx_len < self.max_len:
            #too short, add padding
            indices += [self.vocab.stoi[PAD]] * (self.max_len - idx_len)
            
        #now return indices with the desired embedding 
        if self.embed_mode is None: 
            #no embedding required, return
            return torch.LongTensor(indices)
        else: 
            #embedding needed 
            vectors = self.vocab.vectors
            
            if self.embed_mode == 'token':
                out = torch.zeros([self.max_len, vectors[0].len], dtype=torch.FloatTensor)
                for i, idx in enumerate(indices):
                    out[:, i] = vectors[idx] 
                return out
            else:
                #return one embedding for utterance 
                return None
            
        
        
    def get_label(self, idx):
        '''
        Return labels as a long vector 
        '''
        return torch.LongTensor(self.labels[idx])

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

#Load GLoVe embeddings
embedding_glove = GloVe(name='6B', dim=100)

train_set = CognitiveDataset(train_frame, 
                             max_len=25,
                             vocab_size=2000,
                             vocab=None,
                             embeddings=embedding_glove,
                             embed_mode='token')

val_set = CognitiveDataset(train_frame, 
                             max_len=25,
                             vocab_size=2000,
                             vocab=train_set.vocab,
                             embeddings=embedding_glove,
                             embed_mode='token')

test_set = CognitiveDataset(train_frame, 
                             max_len=25,
                             vocab_size=2000,
                             vocab=train_set.vocab,
                             embeddings=embedding_glove,
                             embed_mode='token')

assert False 

# Routine to generate dataloader

def create_dataloader(dataset, batch_size=32, shuffle=False):
    '''
    Return a dataloader for the specified dataset and batch size.
    Very similar to most CS 598 DLH homework problems.
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

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

def rnn_grid_search():
    
    # Specs for grid search.
    # NOTE: this does not include learning rate, optimizer type, or loss function types.
    params = {'module__hidden_size': [50,100],
              'module__dropout': [0.1, 0.5],
              'batch_size': [32, 64],
              'epochs': [100]}
    
    # ...this is only a bare start...continue.

# Define starter multi-label RNN, plus its loss function and optimizer.
# Start with the settings that gave the best results for the researchers.

mlm_starter_RNN = MultiLabelRNN(2624, embed_size=100, hidden_size=100, dropout=0.1, num_labels=len(SCHEMAS))
loss_func = nn.BCEWithLogitsLoss()  # Hopefully this gives something like a categorical cross-entropy loss
optimizer = nn.Adam(mlm_starter_RNN.parameters(), lr=0.001)  # Using Keras' default learning rate, which is probably what the researchers used

# Stock routine to evaluate initial RNN.
# Taken from HW3. Will replace with skorch functionality.

def eval_rnn(model, val_loader):
    '''
    Evaluate the given RNN using the validation set.
    This is very similar to the eval_model() routine in HW3's RNN assignment.
    
    This model assumes "optimizer" and "loss_func" have already been created.

    Input:
        model: the RNN to be trained
        val_loader: DataLoader for the validation set
    Output: Spearman correlation coefficients for each schema
    '''
    
    model.eval()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    for x, y in val_loader:
        y_hat = model(x)
        y_score = torch.cat((y_score, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
        
    return mean_absolute_error(y_score, y_true)

# Stock routine to train initial RNN.
# Taken from HW3. Will replace with skorch functionality.

def train_rnn(model, train_loader, val_loader, n_epochs=100):
    '''
    Train the given RNN model on the given training set, and
    evaluate it using the validation set.
    This is very similar to the train() routine in HW3's RNN assignment.
    
    This model assumes "optimizer" and "loss_func" have already been created.

    Input:
        model: the RNN to be trained
        train_loader: DataLoader for the training dataset
        val_loader: DataLoader for the validation set
        n_epochs: the number of training epochs
    Output: none
    '''

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            loss = None            # Initialize loss
            optimizer.zero_grad()  # Zero out the gradient
            y_hat = model(x)       # Predict schemas
            # Compute loss
            loss = loss_func(y_hat, y)
            # Back propagation
            loss.backward()
            optimizer.step()
            # Accumulate training loss
            train_loss += loss.item()
            
        # Compute mean train_loss
        train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: training loss {train_loss}')
        # Evaluate model on validation data
        eval_score = eval_rnn(model, val_loader)
        print(f'Validation score: {eval_score}')
        
    
    

###### Ablation study 