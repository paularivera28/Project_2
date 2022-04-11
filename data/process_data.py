#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline
# 
# In this document is the ETL process, where the data is extract, transformed and load

# ### 1. Import libraries


import requests
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


# ### 2. Extract data
# 
# This part loads and merges data


# ### 2. Load Data

def load_data(messages_filepath, categories_filepath):
    
    '''
    load the two data of interest, merges the data frame 
    
    PARAMETERS
    --------------------------------------------------------
    INPUT
    messages_filepath: database containing the messages information
    
    categories_filepath: database containing the categories features
   
    
    OUTPUT
    merged DataFrame of the two input data
    
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])
    return df


# ### 3. Clean Data

def clean_data(df):
    
    '''    
    made the property cleaning 
    splits column categories into separate columns
    changue thes new columns into dummy variables and removes duplicates
    
    INPUT
    df: the merge data computed before
    
    OUTPUT
    clean data base
    
    '''
    
    # create the new columns with the split of the column categories
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    
    # convert new columns to a dummy variable
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1:])
        
    categories =categories.apply(pd.to_numeric)

    
    # drop the category column and then put the new colums into the dataframe
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    
    #drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    # drop null columns and fix binary response
    df = df.drop(['child_alone'],axis=1)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    
    return df


# ### 4. Load

def save_data(df, database_filename):
    
    '''
    Export the final DataFrame to SQL
    
    INPUT
    df: the final datagrame we transform
    database_filename: the new dataframe we want to save
    
    '''
    database_filepath = database_filename
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: python process_data.py '              'disaster_messages.csv disaster_categories.csv '              'DisasterResponse.db')


if __name__ == '__main__':
    main()