# Disaster Response Pipeline Project

### About Project
This repository contains a web app takes any message in a disaster and classifies them according to people need (water, food, medical help, security, shelter, etc.). This web app was designed to help people by classifying disaster messages into categories to direct them to proper aid organisations.
The app uses ETL and ML pipelines to classify disaster messages.

### Explanation of the Files
* app
  * template
    * master.html # main page of web app
    * go.html # classification result page of web app
  * run.py # Flask file that runs app
* data
  * disaster_categories.csv # data to process
  * disaster_messages.csv # data to process
  * process_data.py # ETL pipeline which takes csv files and creates SQL database containing cleaned dataframe
  * DisasterResponse.db # database to save clean data to
  * ETL Pipeline Preparation.ipynb # Jupyter notebook that was used to prepare ETL pipeline file (process_data.py)
* models
  * train_classifier.py # ML pipeline which uses SQL database and build and tune ML model containing pipeline and grid search. It creates pickle file contains saved model
  * classifier.pkl # saved model
  * ML Pipeline Preparation.ipynb # Jupyter notebook that was used to prepare ML pipeline file (train_classifier.py)
* README.md

### How to Run Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command to find web app url
	`env | grep WORK"`

3. Run the following command in the app's directory to run your web app.
    `python app/run.py`

4. Go to http://WORKSPACEID-3001.WORKSPACEDOMAIN

### Screenshots of Web App
![Web app screenshot 1](https://github.com/tbuyar/disaster_response_app/blob/main/web%20app%20screenshot1.JPG "Web app screenshot 1")
![Web app screenshot 2](https://github.com/tbuyar/disaster_response_app/blob/main/web%20app%20screenshot2.JPG "Web app screenshot 2")

*This project is part of the Udacity Data Science Nanodegree program. Some of the codes were implemented by the instructions of the program.
