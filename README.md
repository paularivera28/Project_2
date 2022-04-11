# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [File Description](#Description)
3. [Instructions](#Instructions)
4. [Results](#Results)


## Project Motivation <a name="motivation"></a>
This project is part of the Data Scientist Nanodegree Program of udacity, who has the aim to implement a pipeline to transform and clean the data and create a classification model to analyse several messages and can know the generall idea of the message and give an adecuate response.


## Files Description <a name="Description"></a>

The repositori has the following 3 folders

- data: in this folder you can found the supplies for the project, that is the messages and the categories, also there you can foun the notebook used to transform and clean the data an the process_data.py who was the inprove code from the notebook and finaly DisasterResponse a dbo file with the transformed data.

- model: in this folder cotains the procedure to creat a pipeline for train, build and evaluate a clasiffication model, the model I select is a BaggingClassifier to categorize message this is on the filed train_classifier.py.

- app: in this folder cotains a the scrip run.py with the html templates given in the classroom necesary to run the API.


## Instructions <a name="Instructions"></a>

Follow the steps below to run the app

1. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To change the location to the directory app write `cd app` and then `python run.py` to run the web app

3. Visit this link to see the Dashboard  http://0.0.0.0:3001/

## Results <a name="Results"></a>
Here is an imagen of the resulting app create with all the development sollowed in the project
![image](https://user-images.githubusercontent.com/100497288/162668299-8ac76e57-b0a8-4cca-9a17-234d20ef0f73.png)

