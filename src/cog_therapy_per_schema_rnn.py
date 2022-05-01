""" 
Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Reproduce the per-schema RNNs in the paper below.

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

# Local imports to help make scripts more modular.
import cog_globals as GLOB
import cog_prep_data as prep

### Arguments for RNNs

# Desired embedding level: 'token' or 'utterance'
embed_level = 'token'
# Learning rate
lr = 0.002
# Dropout layer
dropout = 0.1
# Training epochs
n_epochs = 100

### Set seed. Copied from HW3 RNN notebook.

#random.seed(GLOB.seed)
np.random.seed(GLOB.seed)
torch.manual_seed(GLOB.seed)
#os.environ["PYTHONHASHSEED"] = str(GLOB.seed)

###### Model definition

### Per-schema RNN

class PerSchemaRNN(nn.Module):
    '''
    PyTorch implementation of the researchers' per-schema RNN model.
    This is a bidirectional LSTM with a dropout layer and a dense
    output layer, with softmax activation.
    The dense layer spits out 4 outputs, one for each point on the
    rating scale for how well a thought record corresponds to a schema.
    '''

    def __init__(self, vocab_size, embeddings=None, pad_idx=None, hidden_size=GLOB.glove_embed_dim, dropout=dropout, num_label_vals=len(GLOB.RATING_VALS)):
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
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        embeds = self.embedding(x)
        
        lstm_out, (hidden_state_n, cell_state_n) = self.lstm(embeds)
        #get last layer of output 
        lstm_last = lstm_out[:, -1, :].squeeze(1)
                
        do_out = self.do(lstm_last)
        fc_out = self.fc(do_out)
        label_probs = self.softmax(fc_out)
                
        return label_probs

###### Training and evaluation routines

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
    Output: loss: mean absolute error
    '''
    
    model.eval()
    y_score = torch.FloatTensor()
    y_true = torch.FloatTensor()
    for x, y in val_loader:
        # Remove superfluous middle dimension of ground-truth values
        #y = torch.squeeze(y, dim=1)
        y_hat = model(x)
        y_score = torch.cat((y_score, y_hat.detach()), dim=0)
        y_true = torch.cat((y_true, y.detach()), dim=0)
        
    return mean_absolute_error(y_score, y_true)

# Stock routine to train initial RNN.
# Taken from HW3. Will replace with skorch functionality.

def train_rnn(model, train_loader, val_loader, n_epochs=n_epochs):
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
            # Remove superfluous middle dimension of ground-truth values
            #y = torch.squeeze(y, dim=1)
            loss = None            # Initialize loss
            optimizer.zero_grad()  # Zero out the gradient
            y_hat = model(x)       # Predict labels
            
            #print(f"Shape of y: {y.shape}")
            #print(f"Shape of y_hat: {y_hat.shape}")
            #print(f"y: {y}")
            #print(f"y_hat: {y_hat}")
            
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
        
###### Calculate class weights for the loss function

def class_weights(schema, schema_label_vals):
    '''
    Treating utterance schema ratings as classes gives highly imbalanced classes.
    Most utterances are related to one or two schemas and not at all to the rest.
    This means the RNN can be most successful by rating every utterance as being
    unrelated to any schema at all.
    For a given schema, this routine calculates weights to feed the loss function,
    which may help ameliorate the class imbalance problem.
    
    Input: schema: name of schema
        schema_label_vals: ratings of all utterances for a given schema
    Output: array of four weights for each rating (0-3)
    '''
    
    # Compute normalized freqency for each class label.
    rating_freq = schema_label_vals.groupby([schema])[schema].count()
    rating_freq_norm = rating_freq / rating_freq.sum()
    
    # Invert those normalized frequencies to assign higher weights to rare classes.
    neg_norm = 1 - rating_freq_norm
    
    return neg_norm.tolist() / neg_norm.sum()
   
        
###### Load data

in_frame = prep.read_data(process_mode=None)

###### Preprocess data

# Tokenize
in_frame = prep.tokenize(in_frame)

# Split dataset into training / validation / test sets
train_frame, val_frame, test_frame = prep.split_data(in_frame,
                                                    train_fraction=GLOB.train_fraction,
                                                    val_fraction=GLOB.val_fraction,
                                                    test_fraction=GLOB.test_fraction)
###### Load GLoVe embeddings

embedding_glove = GloVe(name='6B', dim=GLOB.glove_embed_dim)

###### Run models ######

# Start timer for training
train_start_time = time.time()

# Initialize lists for goodness-of-fit measures    
# Initialize arrays of corr coefficients and p values
rho_array = np.zeros(len(GLOB.SCHEMAS))
p_array = np.zeros(len(GLOB.SCHEMAS))

### Set weights for the loss function
# Set weights based on the frequency of each rating for each schema.
# For each 



### Loop over schemas. Run model for each schema.

for i, schema in enumerate(GLOB.SCHEMAS):

        print(f'\nTraining model for schema {schema}\n')
        
        ### Get training, validation and test datasets for the chosen schema.
        # Choose dataset class based on the embedding level.
        
        if embed_level == 'token':
            
            train_set = prep.TokenDataset(train_frame, 
                                          schemas=[schema],
                                          max_len=GLOB.max_utt_length,
                                          vocab_size=GLOB.max_vocab_size,
                                          vocab=None,
                                          embeddings=embedding_glove)

            val_set = prep.TokenDataset(val_frame, 
                                        schemas=[schema],
                                        max_len=GLOB.max_utt_length,
                                        vocab_size=GLOB.max_vocab_size,
                                        vocab=train_set.vocab,
                                        embeddings=embedding_glove)

            test_set = prep.TokenDataset(test_frame, 
                                         schemas=[schema],
                                         max_len=GLOB.max_utt_length,
                                         vocab_size=GLOB.max_vocab_size,
                                         vocab=train_set.vocab,
                                         embeddings=embedding_glove)

        elif embed_level == 'utterance':
            
            train_set = prep.UtteranceDataset(train_frame, 
                                              schemas=[schema],
                                              max_len=GLOB.max_utt_length,
                                              vocab_size=GLOB.max_vocab_size,
                                              vocab=None,
                                              embeddings=embedding_glove)
            
            val_set = prep.UtteranceDataset(val_frame, 
                                            schemas=[schema],
                                            max_len=GLOB.max_utt_length,
                                            vocab_size=GLOB.max_vocab_size,
                                            vocab=train_set.vocab,
                                            embeddings=embedding_glove)
            
            test_set = prep.UtteranceDataset(test_frame, 
                                             schemas=[schema],
                                             max_len=GLOB.max_utt_length,
                                             vocab_size=GLOB.max_vocab_size,
                                             vocab=train_set.vocab,
                                             embeddings=embedding_glove)

        ### Create dataloaders for our three datasets
        train_loader = prep.create_dataloader(train_set, shuffle=True)
        val_loader = prep.create_dataloader(val_set, shuffle=False)
        # For test_set, return entire set from dataloader
        test_loader = prep.create_dataloader(test_set, batch_size=len(test_set), shuffle=False)
        
        ### Create model
        model = PerSchemaRNN(vocab_size=len(train_set.vocab), 
                             embeddings=train_set.embed_vec, 
                             pad_idx=train_set.vocab[GLOB.PAD],
                             hidden_size=GLOB.glove_embed_dim, 
                             dropout=dropout, 
                             num_label_vals=len(GLOB.RATING_VALS))
        
        # Set weights for loss function based on rating frequency counts for the schema in the training data.
        rating_weights = class_weights(schema, train_frame[[schema]])
        
        ### Create loss function and optimizer
        # Original Keras model used categorical cross-entropy loss.
        #loss_func = nn.CrossEntropyLoss(weight=torch.tensor(rating_weights))  
        loss_func = nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        ### Train and evaluate model
        # For each epoch, show training loss and validation score (mean absolute error)
        train_rnn(model, train_loader, val_loader)
        
        ### Get test data and predictions
        # Get entire processed test data and labels
        test_x, test_y = next(iter(test_loader))
        #test_y = torch.squeeze(test_y, dim=1)
        # Get test predictions
        test_y_hat = model(test_x)
        
        ### Calculate goodness of fit
        rho, p_val = spearmanr(test_y, test_y_hat.detach().numpy(), axis=None)
        rho_array[i] = rho
        p_array[i] = p_val
        print(f'\n{schema} model Spearman coefficient: {rho}, p-value: {p_val}\n')


##### Display results

# Training time in seconds for training
train_time = time.time() - train_start_time

# Combine goodness of fit measures
test_gof = pd.DataFrame({
                "schema": GLOB.SCHEMAS,
                "estimate": rho_array.tolist(),
                "p": p_array.tolist()})

### Show model goodness of fit

# Display Spearman rank-order correlation for each schema between predicted
# schema correspondences and ground truth.
print("\nPer-Schema RNN Test Set Goodness of Fit (Spearman r) Per Schema:")
print(test_gof.to_string(index=False))

# Display training time in seconds
print(f"\nTraining time: {train_time} seconds")

# Compute and display RAM usage
final_ram_used = psutil.virtual_memory()[3]
script_ram_k_used = (final_ram_used - init_ram_used) / 1024
print(f"\nRAM used by script: {script_ram_k_used} K")


###### Ablation study 