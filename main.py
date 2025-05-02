""" 
Trains random forest model on gut microbiome
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle as pkl

from Bio import SeqIO
from pathlib import Path
from datetime import datetime
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def genus_dict() -> dict[str, list]:
    """ 
    Returns a dictionary with bacteria genus as keys mapped to empty lists.
    """
    ld_dict = {}

    # returns a list of all files in directory
    fa_files = os.listdir(r'C:\Users\artis\OneDrive\Documents\UA_Classes\Spring_2025\ECOL346\metagenome_classification\data\labeled_genome_train_genus')
    
    for file in fa_files:
        label = file.split('.')[0].split('_')[1]
        ld_dict[label] = []
        
    return ld_dict

def genus_seq_df() -> pd.DataFrame:
    """ 
    """
    ld_dict = {
               'Genus': [],
               'Sequence': []
               }
    
    cwd = os.getcwd()

    # returns a list of all files in directory
    fa_files = os.listdir(r'C:\Users\artis\OneDrive\Documents\UA_Classes\Spring_2025\ECOL346\metagenome_classification\data\labeled_genome_train_genus')
    
    # change to labeled data directory
    os.chdir(r'C:\Users\artis\OneDrive\Documents\UA_Classes\Spring_2025\ECOL346\metagenome_classification\data\labeled_genome_train_genus')
    
    for file in fa_files:
        # get genus name
        label = file.split('.')[0].split('_')[1]
        
        # iterate through individual sequences
        for record in SeqIO.parse(file, 'fasta'):
            ld_dict['Sequence'].append(record.seq)
            ld_dict['Genus'].append(label)
            
    # change back to original directory        
    os.chdir(cwd)
    
    # get kmers
    kmers = kmer_space()
    
    # add kmer cols (195100 x 4^4 + 2 = 258)
    df = pd.DataFrame(ld_dict, columns = ['Genus', 'Sequence'] + kmers)
    
    for kmer in kmers:
        df[kmer] = df['Sequence'].apply(lambda seq: (1 if kmer in seq else 0))
    
    return df

def kmer_space() -> list[str]:
    """ 
    :return kmers: list of all possible 4-mers using A, T, G, and C.
    """
    nucleotides = ['A', 'T', 'G', 'C']
    k = 2

    kmers = []

    for first in nucleotides:
        for second in nucleotides:
            for third in nucleotides:
                for fourth in nucleotides:
                    kmers.append(first + second + third + fourth)
                    
    return kmers

# def get_models():
#     """ 
#     """
#     print(f'Training models at {datetime.time.now()}\n')
    
#     ### Constants
#     random_state = 42
    
#     ### Read dataframe
#     start = datetime.datetime.now()
#     df = pd.read_csv('genus_seq_df.csv', index_col=0)
#     end = datetime.datetime.now()
#     print(f'\nDataframe read from csv at {end} after {(end-start).hour} hours and {(end-start).minute} minutes.\n')
    
#     ### Get x and y
#     kmers = kmer_space()
#     X = df[kmers]
#     y = df['Genus']
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     print(f'Finished splitting X and y at {datetime.datetime.now()}.\n')

#     ### Initialize models
#     model = BalancedRandomForestClassifier(random_state=random_state)
#     selector = SequentialFeatureSelector(model, cv=3) # runs forward selection w 5 fold cv by default
#     print(f'Models initialized at {datetime.datetime.now()}\n')
    
#     ### Feature selection
#     start = datetime.datetime.now()
#     print(f'Fitting feature selector at {start}.\n')
#     selector.fit(x_train, y_train)
#     end = datetime.datetime.now()
#     print(f'Finished fitting feature selector at {end} after {(end-start).hours} hours and {(end-start).minute} minutes.\n')
    
#     selected_x_train = selector.transform(x_train)
#     print(f'Applied feature selection at {datetime.datetime.now()}\n')
    
#     ### Fit model
#     start = datetime.datetime.now()
#     print(f'Fitting forest at {start}.\n')
#     model.fit(selected_x_train, y_train)
#     end = datetime.datetime.now()
#     print(f'Finished fitting forest at {end} after {(end-start).hours} hours and {(end-start).minute} minutes.\n')
    
#     return selector, model

# commented function above was written from scratch, function below is the result
# after asking ChatGPT for a docstring (edited later by me) and to check for 
# minor syntax errors/mistypes
    
def get_models(csv_path='genus_seq_df.csv'):
    """
    Trains a Balanced Random Forest classifier with sequential feature selection.

    This function performs the following steps:
        1. Loads a CSV file containing k-mer features and genus labels.
        2. Extracts feature columns based on the `kmer_space()` function and splits the data into training and testing sets.
        3. Initializes a `BalancedRandomForestClassifier` and a `SequentialFeatureSelector`.
        4. Fits the feature selector on the training data.
        5. Transforms the training data using the selected features.
        6. Trains the classifier on the selected features of the training set.
        7. Logs the timing and progress of each step for transparency.

    :param csv_path (str): Path to the CSV file containing the data. Defaults to 'genus_seq_df.csv'.


    :return selector (SequentialFeatureSelector): The fitted feature selector.
    :return    model (BalancedRandomForestClassifier): The trained classification model.
    :return    x_test (pd.DataFrame): Test feature set for later evaluation.
    :return     y_test (pd.Series): Test labels for later evaluation.
    :return     X_train:
    :return     y_train:
    """
    print(f'Training models at {datetime.now()}\n')

    ### Constants
    random_state = 42

    ### Read dataframe
    start = datetime.now()
    df = pd.read_csv(csv_path, index_col=0)
    end = datetime.now()
    elapsed = end - start
    print(f'\nDataframe read from csv at {end} after {elapsed.seconds // 3600} hours and {(elapsed.seconds % 3600) // 60} minutes.\n')

    ### Get X and y
    kmers = kmer_space()
    X = df[kmers]
    y = df['Genus']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(f'Finished splitting X and y at {datetime.now()}.\n')

    ### Initialize models
    model = BalancedRandomForestClassifier(random_state=random_state)
    selector = SequentialFeatureSelector(model, cv=3)
    print(f'Models initialized at {datetime.now()}\n')

    ### Feature selection
    start = datetime.now()
    print(f'Fitting feature selector at {start}.\n')
    selector.fit(x_train, y_train)
    end = datetime.now()
    elapsed = end - start
    print(f'Finished fitting feature selector at {end} after {elapsed.seconds // 3600} hours and {(elapsed.seconds % 3600) // 60} minutes.\n')

    selected_x_train = selector.transform(x_train)
    print(f'Applied feature selection at {datetime.now()}\n')

    ### Fit model
    start = datetime.now()
    print(f'Fitting forest at {start}.\n')
    model.fit(selected_x_train, y_train)
    end = datetime.now()
    elapsed = end - start
    print(f'Finished fitting forest at {end} after {elapsed.seconds // 3600} hours and {(elapsed.seconds % 3600) // 60} minutes.\n')

    return selector, model, x_test, y_test, x_train, y_train
    

def main():
    print(f'\nMain started at {datetime.now()}\n')
    ## Ensure we're working from the parent directory to avoid 'File Not Found' errors
    os.chdir(r'C:\Users\artis\OneDrive\Documents\UA_Classes\Spring_2025\ECOL346\metagenome_classification')
    
    fname = 'metagenome_classifier.pkl'
    try:
        if not os.path.exists(fname):
            selector, model, x_test, y_test, x_train, y_train = get_models()
            pkl.dump(model, open(fname, 'wb'))
            pkl.dump(selector, open('metagenome_feature_selector.pkl', 'wb'))
            pkl.dump(y_test, open('metagenome_y_test.pkl', 'wb'))
            pkl.dump(x_test, open('metagenome_x_test.pkl', 'wb'))
            print(f'Pickled model at {datetime.now()}\n')
        else:
            model = pkl.load(open(fname, 'rb'))
            selector = pkl.load(open('metagenome_feature_selector.pkl', 'rb'))
            y_test = pkl.load(open('metagenome_y_test.pkl', 'rb'))
            x_test = pkl.load(open('metagenome_x_test.pkl', 'rb'))
            print(f'Read pickled model at {datetime.now()}') 
            
    except Exception as e:
        print('Issue in main, lines 219-232. Pickling went sour :(\n')
        print(e)
        return
    
    # feature select test data
    x_test_selected = selector.transform(x_test) 
    
    # predict on test data       
    y_pred = model.predict(x_test_selected)
    
    # write report with timestamp
    report = classification_report(y_test, y_pred)
    report_file = 'metagenome_report.txt'
    timestamp = f"\n\nReport generated at: {datetime.now()}\n"
    with open(report_file, 'a') as f:       # appends data instead of overwriting
        f.write(report)
        f.write(timestamp)
    
if __name__ == '__main__':
    main()
