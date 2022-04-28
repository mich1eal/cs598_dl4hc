""" 
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

# To track RAM and CPU usage
import time
import psutil
# Store initial RAM usage to help tare final usage
init_ram_used = psutil.virtual_memory()[3]

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torchtext.vocab import GloVe
# To help perform hyperparameter grid search
#from skorch import NeuralNetClassifier
#from sklearn.model_selection import GridSearchCV
# To compute model score on validation set
from sklearn.metrics import mean_absolute_error
# To compute model goodness-of-fit
from scipy.stats import spearmanr
# To compute distance in kNN model
#from scipy.spatial.distance import cosine

# Local imports to help make scripts more modular.
import cog_globals as GLOB
import cog_prep_data as prep

# Set seed. Copied from HW3 RNN notebook.
#random.seed(GLOB.seed)
np.random.seed(GLOB.seed)
torch.manual_seed(GLOB.seed)
#os.environ["PYTHONHASHSEED"] = str(GLOB.seed)

###### Load data
in_frame = prep.read_data(preprocessed=False)

###### Preprocess data
in_frame = prep.tokenize(in_frame)

# Split dataset into training / validation / test sets
train_frame, val_frame, test_frame = prep.split_data(in_frame,
                                                    train_fraction=GLOB.train_fraction,
                                                    val_fraction=GLOB.val_fraction,
                                                    test_fraction=GLOB.test_fraction)
###### Prepare data
#Load GLoVe embeddings
embedding_glove = GloVe(name='6B', dim=GLOB.glove_embed_dim)

train_set = prep.TokenDataset(train_frame, 
                             max_len=GLOB.max_utt_length,
                             vocab_size=GLOB.max_vocab_size,
                             vocab=None,
                             embeddings=embedding_glove)

train_set_utterance = prep.UtteranceDataset(train_frame, 
                                            max_len=GLOB.max_utt_length,
                                            vocab_size=GLOB.max_vocab_size,
                                            vocab=None,
                                            embeddings=embedding_glove)

val_set = prep.TokenDataset(val_frame, 
                             max_len=GLOB.max_utt_length,
                             vocab_size=GLOB.max_vocab_size,
                             vocab=train_set.vocab,
                             embeddings=embedding_glove)

test_set = prep.TokenDataset(test_frame, 
                             max_len=GLOB.max_utt_length,
                             vocab_size=GLOB.max_vocab_size,
                             vocab=train_set.vocab,
                             embeddings=embedding_glove)

# Create dataloaders for our three datasets
train_loader = prep.create_dataloader(train_set, shuffle=True)
val_loader = prep.create_dataloader(val_set, shuffle=False)
# For test_set, return entire set from dataloader
test_loader = prep.create_dataloader(test_set, batch_size=len(test_set), shuffle=False)

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
        rho: pandas dataframe with three columns:
                "schema" for schema name,
                "estimate" for r values, and
                "p" for p-values.
    '''
    
    # Initialize arrays of corr coefficients and p values
    rho_array = np.zeros(X.shape[1])
    p_array = np.zeros(X.shape[1])

    # Compute coefficient over each schema and save
    for schema in range(len(GLOB.SCHEMAS)):
        rho, p_val = spearmanr(X[:, schema], Y[:, schema])
        rho_array[schema] = rho
        p_array[schema] = p_val

    return pd.DataFrame({
        "schema": GLOB.SCHEMAS,
        "estimate": rho_array.tolist(),
        "p": p_array.tolist()})

###### Baseline models

### kNNs



###### Paper primary model

### Multi-label RNN

class MultiLabelRNN(nn.Module):
    
    '''
    PyTorch implementation of the researchers' multi-label RNN.
    This is a bidirectional LSTM with a dropout layer and a dense
    output layer, with sigmoid activation.
    '''
    
    def __init__(self, vocab_size, embeddings=None, pad_idx=None, hidden_size=GLOB.glove_embed_dim, dropout=0.5, num_labels=len(GLOB.SCHEMAS)):
        super().__init__()
                
        if embeddings is not None:
            embed_size = len(embeddings[0])
            self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=pad_idx, freeze=True)
        else:
            embed_size = GLOB.glove_embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear((2 * hidden_size), num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embeds = self.embedding(x)
        
        lstm_out, (hidden_state_n, cell_state_n) = self.lstm(embeds)
        #get last layer of output 
        lstm_last = lstm_out[:, -1, :].squeeze(1)
                
        do_out = self.do(lstm_last)
        label_probs = self.sigmoid(self.fc(do_out))
                
        return label_probs

### Per-schema RNN

class PerSchemaRNN(nn.Module):
    '''
    PyTorch implementation of the researchers' per-schema RNN model.
    This is a bidirectional LSTM with a dropout layer and a dense
    output layer, with softmax activation.
    The dense layer spits out 4 outputs, one for each point on the
    rating scale for how well a thought record corresponds to a schema.
    '''

    def __init__(self, vocab_size, embeddings=None, pad_idx=None, hidden_size=GLOB.glove_embed_dim, dropout=0.5, num_label_vals=4):
        super().__init__()
                
        if embeddings is not None:
            embed_size = len(embeddings[0])
            self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=pad_idx, freeze=True)
        else:
            embed_size = GLOB.glove_embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear((2 * hidden_size), num_label_vals)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        embeds = self.embedding(x)
        
        lstm_out, (hidden_state_n, cell_state_n) = self.lstm(embeds)
        #get last layer of output 
        lstm_last = lstm_out[:, -1, :].squeeze(1)
                
        do_out = self.do(lstm_last)
        label_probs = self.softmax(self.fc(do_out))
                
        return label_probs

# Define starter multi-label RNN, plus its loss function and optimizer.
# Start with the settings that gave the best results for the researchers.

mlm_starter_RNN = MultiLabelRNN(vocab_size=len(train_set.vocab), 
                                embeddings=train_set.embed_vec, 
                                pad_idx=train_set.vocab[GLOB.PAD],
                                hidden_size=GLOB.glove_embed_dim, 
                                dropout=0.1, 
                                num_labels=len(GLOB.SCHEMAS))
# Original Keras model used categorical cross-entropy loss.
loss_func = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(mlm_starter_RNN.parameters(), lr=0.002)

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
    y_score = torch.FloatTensor()
    y_true = torch.FloatTensor()
    for x, y in val_loader:
        y_hat = model(x)
        y_score = torch.cat((y_score, y_hat.detach()), dim=0)
        y_true = torch.cat((y_true, y.detach()), dim=0)
        
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
        
### Train models

# Start timer for training
train_start_time = time.time()

# For each epoch, show training loss and validation score (mean absolute error)
train_rnn(mlm_starter_RNN, train_loader, val_loader)

# Training time in seconds for training
train_time = time.time() - train_start_time

### Show model goodness of fit

# Display Spearman rank-order correlation for each schema between predicted
# schema correspondences and ground truth.
# Use test set.

# Get entire processed test data and labels
test_x, test_y = next(iter(test_loader))
# Get test predictions
test_y_hat = mlm_starter_RNN(test_x)
    
# Get and display goodness of fit for each schema
test_gof = spearman_r(test_y, test_y_hat.detach().numpy())
print("\nRNN Multi-Label Model Test Set Goodness of Fit (Spearman r) Per Schema:")
print(test_gof.to_string(index=False))

# Display training time in seconds
print(f"\nTraining time: {train_time} seconds")

# Compute and display RAM usage
final_ram_used = psutil.virtual_memory()[3]
script_ram_k_used = (final_ram_used - init_ram_used) / 1024
print(f"\nRAM used by script: {script_ram_k_used} K")


###### Ablation study 