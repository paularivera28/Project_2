#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline
# 
# In this document is the Machine Learning process, where the final data is imported an a machine learning model is created

# ## 1. Import Libraries
# 
# Import needed libraries

# In[ ]:


import sys
import re
import os
import numpy as np
import pandas as pd
import requests
import sqlite3
import pickle


from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score



# ### 1. Load Data


def load_data(database_filepath):

    '''
    load_data loads the final data that comes from process_data.py and creates X and y DataFrames
    
    INPUT
    database_filepath: Path of database saved by process_data.py
    
    OUTPUT
    X DataFrame with messages
    y DataFrame with categories
     
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # create X, y
    X = df['message']
    y = df[df.columns[4:]]  
    category_names = y.columns
    
    return X, y, category_names


# ### 3.  Write a tokenization function to process your text data


def tokenize(text):
    
    """Tokenize the text 
    
    Tokenize the text information by using word_tokenize and WordNetLemmatizer
    
    INPOUT:
    -----------
    text: the information of the message in the data
    
    OUTPUT:
    ----------
    result: the modified text
    """
    
    #To hadle with special characteres
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #  group the different forms of a word
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 4. Model Building

def build_model():
    
    '''
    pipeline for modelling with MultiOutputClassifier
    
    OUTPUT
    model/pipeline
    
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(BaggingClassifier()))
    ])
    
    return pipeline

#############################################################################
## uncoment this part if you want to create the model with GridSearchCV
## This option will take more time than the oder option
##############################################################################
#def build_model():
    
#    '''
#    pipeline for modelling with MultiOutputClassifier
    
#    OUTPUT
#    model/pipeline
    
#    '''
        
    # params
#    params = {
#    'base_estimator__max_depth' : [1, 2, 3, 4, 5],
#    'max_samples' : [0.05, 0.1, 0.2, 0.5]
#     }

 #   # train model
 #   cv = GridSearchCV(pipeline, param_grid=params, cv=3)
 #   cv = pipeline
 #   return pipeline


# ### 5. Model Evaluation


def evaluate_model(model, X_test, y_test, category_names):
    """
    metrics to evaluate the model
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred, target_names = category_names))


# ### 6. Save Model
# 


def save_model(model, model_filepath):
    """
    generate the pickle to save the results
    
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()