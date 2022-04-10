#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries 
# Here we are going to upload the necesary library
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[20]:


# import libraries
import requests
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


# ### 2. Load datasets.
# 
# This part loads and merges the data necesary for the project

# In[21]:


# load messages dataset
messages = pd.read_csv(r"messages.csv")
messages.head()


# In[22]:


# load categories dataset
categories = pd.read_csv(r"categories.csv")
categories.head()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[23]:


# merge datasets
df = messages.merge(categories, on="id")
df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[43]:


# create a dataframe of the 36 individual category columns
categories = df.categories.str.split(";", expand=True)
categories.head()


# In[25]:


# select the first row of the categories dataframe
row = categories.iloc[0, :]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x[:-2])
print(category_colnames)


# In[26]:


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[45]:


for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x.split("-")[1])
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
categories.head()


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[46]:


# drop the original categories column from `df`

df.drop('categories', axis=1, inplace=True)
df.head()


# In[47]:


# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
df.head()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[50]:


# check number of duplicates
#the number of total duplicates of the dataset
check_dup = duplicated().sum()
check_dup


# In[51]:


# check number of duplicates in the id column
#the number of total duplicates of the dataset by
num_dup_id = df.duplicated(subset='id', keep='first').sum() 
num_dup_id


# In[52]:


# drop duplicates
df.drop_duplicates(subset='id', inplace=True)


# In[53]:


# check number of duplicates before drop
check_dup = duplicated().sum()
check_dup


# In[54]:


# check number of duplicates in the id column
#the number of total duplicates of the dataset by id  before drop
num_dup_id = df.duplicated(subset='id', keep='first').sum()
num_dup_id


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[57]:


engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('DisasterResponse', engine, index=False)


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:




