# importing libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
import sqlite3

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM data', engine)
    X = df["message"]
    y = df.drop(columns=["id","message", "original", "genre"])
    print(X.shape, y.shape, y.columns.values)
    return X, y, y.columns.values

def tokenize(text):
    
    """ 
    tokenization function to process text data
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

def build_model():
    
    
    # pipeline should take in the message column as input and output classification results on the other 36 categories in the dataset
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    # Grid search parameters and model
    parameters = {
              "tfidf__use_idf": (True, False)
                 }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    #prediction
    Y_pred = model.predict(X_test)
    
    # Report the f1 score, precision and recall for each output category of the dataset
    for i in range(len(category_names)):
        print(category_names[i],'\n', classification_report(Y_test.values[:,i], Y_pred[:,i]))
    
    # Total accuracy
    av_accuracy=(Y_pred == Y_test).mean().mean()
    print("Avarage accuracy: ", round(av_accuracy*100,0),"%")


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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