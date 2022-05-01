""" 
Michael Miller and Kurt Tuohy
CS 598 Deep Learning for Healthcare - University of Illinois 
Final project - Paper Results Verification
4/4/2022

Functions for preprocessing data

Paper: "Natural language processing for cognitive therapy: Extracting schemas from thought records"
by Franziska Burger, Mark A. Neerincx, and Willem-Paul Brinkman
DOI: https://doi.org/10.1371/journal.pone.0257832
Data repo: https://github.com/mich1eal/cs598_dl4hc
For dependencies, and data acquisition instructions, please see this repository's readme
"""

import cog_globals as GLOB
import pandas as pd
from autocorrect import Speller
spell = Speller(lang='en')

#Everything except'.,!?-
remove_punct = """"#$%&()*+-/:;<=>@[\]^_`{|}~"""



def clean(row):
    '''
    Make sentence lowercase, strip extra whitespace, run spell check
    '''
    text = row['Reply']
    text = text.lower().strip()
    
    #Remove punctuation except '.,!?-
    text = text.translate(str.maketrans('', '', remove_punct))

    #Add end of sentence punctuation
    if text[-1] not in ['.', '?']:
        text = text + '.'
    
    corrected = spell(text)
    
    return corrected
      
# use the authors raw data 
in_file = '{}/CoreData.csv'.format(GLOB.DATA_DIR)
out_file = '{}/{}'.format(GLOB.DATA_DIR, GLOB.CUSTOM_PREPROCESS)

raw_frame = pd.read_csv(in_file, sep=';', header=0)

#discard rows that the authors excluded because they don't have enough data
raw_frame = raw_frame[raw_frame['Exclude'] == 0]

#clean each line
raw_frame['Utterance'] = raw_frame.apply(lambda row: clean(row),axis=1)

#write back to outfile
raw_frame.to_csv(out_file, index=False)