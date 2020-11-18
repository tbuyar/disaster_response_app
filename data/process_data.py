# importing libraries
import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """ 
    load data from csv file and return dataframe
  
    Parameters: 
    messages_filepath (str): file path of messages.csv file
    categories_filepath (str): file path of categories.csv file
  
    Returns: 
    DataFrame: merged dataframe from two csv files
  
    """
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")

    return df
    
def clean_data(df):
    """ 
    clean dataframe by several steps which includes splitting columns, converting dtypes, dropping duplicates
  
    Parameters: 
    df (DataFrame): merged dataframe
  
    Returns: 
    DataFrame: cleaned and organized dataframe
  
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(expand=True, pat=";")

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.str.split(pat="-",expand=True).iloc[:,0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str.split(pat="-",expand=True).iloc[:,1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype("int64")
        
    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # drop the rows that related=2
    df=df[df["related"]!=2]
   
    return df
    
def save_data(df, database_filename):
    """ 
    save the clean dataset into an sqlite database
  
    Parameters: 
    df (DataFrame): final dataframe
    database_filename (str): name of database
    
    Returns: 
    None
  
    """
    # create engine and save it as SQL database by using pandas to_sql function
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()