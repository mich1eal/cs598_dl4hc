""" 
Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Reproduce the k-Nearest Neighbors models in the paper below.
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
# kNN models
from sklearn.neighbors import NearestNeighbors
# To compute distance in kNN model
from scipy.spatial import distance
# To help perform hyperparameter grid search
#from skorch import NeuralNetClassifier
#from sklearn.model_selection import GridSearchCV
# Statistical mode for kNN classifier
from scipy.stats import mode
# To compute model goodness-of-fit
from scipy.stats import spearmanr

import warnings

# Local imports to help make scripts more modular.
import cog_globals as GLOB
import cog_prep_data as prep

### Arguments for kNNs

# Desired embedding level: 'token' or 'utterance'
embed_level = 'utterance'
# Desired values of k for classification and regression models.
# Using the paper's best-performing settings
classification_k = 4
regression_k = 5

### Set seed. Copied from HW3 RNN notebook.

#random.seed(GLOB.seed)
np.random.seed(GLOB.seed)
torch.manual_seed(GLOB.seed)
#os.environ["PYTHONHASHSEED"] = str(GLOB.seed)

###### Model definition

### Distance metric for kNN

def cosine_dist(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return distance.cosine(x,y)

### kNN classification

class kNNClassification():
    '''
    Implementation of the researchers' kNN classifier.
    Since the paper used multi-class labels, they used
    custom evaluation logic rather than using the basic
    sklearn KNeighborsClassifier.
    '''

    def __init__(self, k, distance_metric):
        self.k = k
        self.metric = distance_metric
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
    
    def fit(self, train_x):
        self.model.fit(train_x)
        
    def test(self, test_x, train_y, test_y):
        '''
        For the given test utterances, identify the nearest neighbors
        from the training set. Take the statistical mode of the class
        labels from those neighbors.
        '''
        # Array to store all predicted class labels
        test_y_hat = np.zeros(test_y.shape)
        
        # Loop over test utterances
        for i, utterance in enumerate(test_x):
            #print(f"utterance: {utterance}")
            # Predict k nearest training neighbors of the given utterance
            neighbors = self.model.kneighbors([utterance], self.k, return_distance=False)[0]
            #print(f'neighbors: {neighbors}')
            # Array to store predicted neighbors' class labels
            k_pred_class = np.zeros((self.k, test_y.shape[1]))

            # Get class labels of nearest training neighbors
            for j, neighbor in enumerate(neighbors):
                k_pred_class[j,:] = train_y[neighbor,:]
                
            # Take statistical mode of the predicted neighbors' class labels.
            # This gives predicted class labels for the given utterance.
            #print(f'k_pred_classes for neighbors: {k_pred_class}')
            test_y_hat[i,:] = mode(k_pred_class, nan_policy='omit')[0]
            #print(f'Mode of k_pred_classes: {test_y_hat[i,:]}')
            
        return test_y_hat

### kNN Regression

class kNNRegression():
    '''
    Implementation of the researchers' kNN regression model.
    Since the paper used multi-class labels, they used
    custom evaluation logic rather than using the basic
    sklearn KNeighborsRegressor.
    '''

    def __init__(self, k, distance_metric):
        self.k = k
        self.metric = distance_metric
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
    
    def fit(self, train_x):
        self.model.fit(train_x)
        
    def test(self, test_x, train_y, test_y):
        '''
        For the given test utterances, identify the nearest neighbors
        from the training set. Take the average of the class
        labels from those neighbors.
        '''
        # Array to store all predicted class labels
        test_y_hat = np.zeros(test_y.shape)
        
        # Loop over test utterances
        for i, utterance in enumerate(test_x):
            #print(f"utterance: {utterance}")
            # Predict k nearest training neighbors of the given utterance
            neighbors = self.model.kneighbors([utterance], self.k, return_distance=False)[0]
            #print(f'neighbors: {neighbors}')
            # Array to store predicted neighbors' class labels
            k_pred_class = np.zeros(test_y.shape[1])

            # Sum class labels of nearest training neighbors
            for neighbor in neighbors:
                k_pred_class += train_y[neighbor,:]
                
            #print(f'k_pred_classes for neighbors: {k_pred_class}')
            # Take average of class labels.
            test_y_hat[i,:] = np.divide(k_pred_class, self.k)
            #print(f'Average of k_pred_classes: {test_y_hat[i,:]}')
                
        return test_y_hat


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
        # print(f'Spearman correlation for schema {schema}:')
        # true_vs_pred_labels = pd.DataFrame({
        #     'true': X[:, schema],
        #     'pred': Y[:, schema]})
        # print(true_vs_pred_labels.to_string(index=False))
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
# Note that the validation set is only needed to find the best value of k,
# which we're not doing here. Instead we use the researchers' best k values.

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
# For test_set, return entire set from dataloader
test_loader = prep.create_dataloader(test_set, batch_size=len(test_set), shuffle=False)

# Get data from loaders

train_x, train_y = next(iter(train_loader))
val_x, val_y = next(iter(val_loader))
test_x, test_y = next(iter(test_loader))

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

# kNN classifier
knn_classifier = kNNClassification(k=classification_k, distance_metric=cosine_dist)
knn_classifier.fit(train_x)
classifier_test_y_hat = knn_classifier.test(test_x, train_y, test_y)

# kNN regressor
knn_regressor = kNNRegression(k=regression_k, distance_metric=cosine_dist)
knn_regressor.fit(train_x)
regressor_test_y_hat = knn_regressor.test(test_x, train_y, test_y)


# Training & testing time in seconds for training
train_time = time.time() - train_start_time

# Get and display goodness of fit for each schema

classifier_test_gof = spearman_r(test_y, classifier_test_y_hat)
print("\nkNN Classification Test Set Goodness of Fit (Spearman r) Per Schema:")
print(classifier_test_gof.to_string(index=False))

regressor_test_gof = spearman_r(test_y, regressor_test_y_hat)
print("\nkNN Regression Test Set Goodness of Fit (Spearman r) Per Schema:")
print(regressor_test_gof.to_string(index=False))

# Display training time in seconds
print(f"\nTraining + Test time: {train_time} seconds")

# Compute and display RAM usage
final_ram_used = psutil.virtual_memory()[3]
script_ram_k_used = (final_ram_used - init_ram_used) / 1024
print(f"\nRAM used by script: {script_ram_k_used} K")
