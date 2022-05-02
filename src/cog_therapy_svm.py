""" 
Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Reproduce the support vector machine models in the paper below.
Do both classification and regression.

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
from torchtext.vocab import GloVe
# SVM models
from sklearn import svm
# To scale features to have zero mean and unit variance
from sklearn.preprocessing import StandardScaler
# To help perform hyperparameter grid search
#from skorch import NeuralNetClassifier
#from sklearn.model_selection import GridSearchCV
# To compute model goodness-of-fit
from scipy.stats import spearmanr

# Local imports to help make scripts more modular.
import cog_globals as GLOB
import cog_prep_data as prep

### Arguments for kNNs

# Desired embedding level: 'token' or 'utterance'
embed_level = 'utterance'
# SVM kernels to use for classification and regression models.
# Using the paper's best-performing settings
classification_kernel = 'rbf'
regression_kernel = 'rbf'

### Set seed. Copied from HW3 RNN notebook.

#random.seed(GLOB.seed)
np.random.seed(GLOB.seed)
torch.manual_seed(GLOB.seed)
#os.environ["PYTHONHASHSEED"] = str(GLOB.seed)

###### Model definition

### Routine to help scale data to have zero mean and unit variance

def text_scaler(x):
    scaler_texts = StandardScaler()
    scaler_texts = scaler_texts.fit(x)
    return scaler_texts

### SVM classification

class SVMClassification():
    '''
    Implementation of the researchers' SVM classifier.
    These models handle one schema at a time.
    '''

    def __init__(self, kernel, text_scaler):
        self.kernel = kernel
        self.model = svm.SVC(kernel=kernel)
        self.text_scaler = text_scaler
    
    def fit(self, train_x, train_y):
        # Normalize training data
        train_x = self.text_scaler.transform(train_x)
        # Fit model to a single schema's ratings
        self.model.fit(train_x, train_y)
        
    def test(self, test_x):
        '''
        For the given test utterances, predict ratings for a schema
        '''
        # Normalize test data
        text_x = self.text_scaler.transform(test_x)
        return self.model.predict(text_x)

### SVM Regression

class SVMRegression():
    '''
    Implementation of the researchers' SVM regressor.
    These models handle one schema at a time.
    '''

    def __init__(self, kernel, text_scaler):
        self.kernel = kernel
        self.model = svm.SVR(kernel=kernel)
        self.text_scaler = text_scaler
    
    def fit(self, train_x, train_y):
        # Normalize training data
        train_x = self.text_scaler.transform(train_x)
        # Fit model to a single schema's ratings
        self.model.fit(train_x, train_y)
        
    def test(self, test_x):
        '''
        For the given test utterances, predict ratings for a schema
        '''
        # Normalize test data
        text_x = self.text_scaler.transform(test_x)
        return self.model.predict(text_x)

###### Evaluation routine

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

        
###### Load data

in_frame = prep.read_data(process_mode=None)

###### Preprocess data

# Tokenize
in_frame = prep.tokenize(in_frame)

# Split dataset into training / validation / test sets.
# Note that the validation set is only needed to find the best SVM kernel
# which we're not doing here. Instead we use the researchers' best kernel setting.

train_frame, val_frame, test_frame = prep.split_data(in_frame,
                                                    train_fraction=GLOB.train_fraction,
                                                    val_fraction=GLOB.val_fraction,
                                                    test_fraction=GLOB.test_fraction)

###### Prepare data

# Load GLoVe embeddings
embedding_glove = GloVe(name='6B', dim=GLOB.glove_embed_dim)

# Create training, validation and test datasets

### Get training, validation and test datasets for the chosen schema.
# Choose dataset class based on the embedding level.

if embed_level == 'token':
    
    train_set = prep.TokenDataset(train_frame, 
                                  schemas=GLOB.SCHEMAS,
                                  max_len=GLOB.max_utt_length,
                                  vocab_size=GLOB.max_vocab_size,
                                  vocab=None,
                                  embeddings=embedding_glove)

    val_set = prep.TokenDataset(val_frame, 
                                schemas=GLOB.SCHEMAS,
                                max_len=GLOB.max_utt_length,
                                vocab_size=GLOB.max_vocab_size,
                                vocab=train_set.vocab,
                                embeddings=embedding_glove)

    test_set = prep.TokenDataset(test_frame, 
                                 schemas=GLOB.SCHEMAS,
                                 max_len=GLOB.max_utt_length,
                                 vocab_size=GLOB.max_vocab_size,
                                 vocab=train_set.vocab,
                                 embeddings=embedding_glove)

elif embed_level == 'utterance':
    
    train_set = prep.UtteranceDataset(train_frame, 
                                      schemas=GLOB.SCHEMAS,
                                      max_len=GLOB.max_utt_length,
                                      vocab_size=GLOB.max_vocab_size,
                                      vocab=None,
                                      tfidf_tokenizer=None,
                                      embeddings=embedding_glove)
    
    val_set = prep.UtteranceDataset(val_frame, 
                                    schemas=GLOB.SCHEMAS,
                                    max_len=GLOB.max_utt_length,
                                    vocab_size=GLOB.max_vocab_size,
                                    vocab=train_set.vocab,
                                    tfidf_tokenizer=train_set.tfidf_tokenizer,
                                    embeddings=embedding_glove)
    
    test_set = prep.UtteranceDataset(test_frame, 
                                     schemas=GLOB.SCHEMAS,
                                     max_len=GLOB.max_utt_length,
                                     vocab_size=GLOB.max_vocab_size,
                                     vocab=train_set.vocab,
                                     tfidf_tokenizer=train_set.tfidf_tokenizer,
                                     embeddings=embedding_glove)


# Create dataloaders for our three datasets.

# For classical models, return all data at once. No batching needed
train_loader = prep.create_dataloader(train_set, batch_size=len(train_set), shuffle=True)
val_loader = prep.create_dataloader(val_set, batch_size=len(val_set), shuffle=False)
test_loader = prep.create_dataloader(test_set, batch_size=len(test_set), shuffle=False)

# Get data from loaders

train_x, train_y = next(iter(train_loader))
val_x, val_y = next(iter(val_loader))
test_x, test_y = next(iter(test_loader))

# Create text scaler based on training utterances
scaler_texts = text_scaler(train_x)

# Convert datasets to numpy arrays

train_x = train_x.numpy()
train_y = train_y.numpy()
val_x = val_x.numpy()
val_y = val_y.numpy()
test_x = test_x.numpy()
test_y = test_y.numpy()

###### Run models ######

# Start timer for training and testing
train_start_time = time.time()

### Loop over schemas.
# Train and test separate models per schema.

# Numpy arrays to store predictions for each model type
svc_y_hat = np.zeros(test_y.shape)
svr_y_hat = np.zeros(test_y.shape)

for i, schema in enumerate(GLOB.SCHEMAS):
    
    # Initialize models
    svc_model = SVMClassification(kernel=classification_kernel, text_scaler=scaler_texts)
    svr_model = SVMRegression(kernel=regression_kernel, text_scaler=scaler_texts)
    
    # Fit models to schema's training data
    svc_model.fit(train_x, train_y[:, i])
    svr_model.fit(train_x, train_y[:, i])
    
    # Generate predictions
    svc_y_hat[:, i] = svc_model.test(test_x)
    svr_y_hat[:, i] = svr_model.test(test_x)
    
###### Compute and display results

# Training & testing time in seconds for training
train_time = time.time() - train_start_time

# Get and display goodness of fit for each schema

svc_test_gof = spearman_r(test_y, svc_y_hat)
print("\nSVM Classification Test Set Goodness of Fit (Spearman r) Per Schema:")
print(svc_test_gof.to_string(index=False))

svr_test_gof = spearman_r(test_y, svr_y_hat)
print("\nSVM Regression Test Set Goodness of Fit (Spearman r) Per Schema:")
print(svr_test_gof.to_string(index=False))

# Display training and test time in seconds
print(f"\nTraining + Test time: {train_time} seconds")

# Compute and display RAM usage
final_ram_used = psutil.virtual_memory()[3]
script_ram_k_used = (final_ram_used - init_ram_used) / 1024
print(f"\nRAM used by script: {script_ram_k_used} K")
