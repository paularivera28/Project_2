# Disaster Response Pipeline Project

## Table of Contents

1. [Libraries](#Libraries)
2. [Project Motivation](#motivation)
3. [File Description](#Description)
4. [Instructions](#Instructions)
5. [Results](#Results)


## Libraries <a name="Libraries"></a>

In this project I use the following libraries in python:

- Pandas
- Numpy
- math
- json
- matplotlib
- sklearn

## Project Motivation <a name="motivation"></a>

This project is part of the Data Scientist Nanodegree Program of udacity, who has the aim to put in action all the sill we learn in the program, the idea is to understand the behavior of the clients based on thei characteristics and the interactions with the type of offert they recived in the final transaction or purshase by different media like web, social or mobile with the objective of determine the features that make and ofert succesful.

## Files Description <a name="Description"></a>

The repositori has the following 3 folders

- portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)

- profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income

-transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record


## Instructions <a name="Instructions"></a>

Follow the steps below to run the app

1. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To change the location to the directory app write `cd app` and then `python run.py` to run the web app

3. Visit this link to see the Dashboard  http://0.0.0.0:3001/

## Results <a name="Results"></a>

The blog with the result of the analisis is in this link  [post](https://paularivera288.wixsite.com/website/post/exploring-the-seattle-airbnb-data) 




![image](https://user-images.githubusercontent.com/100497288/162784170-dd551273-df7c-476f-8937-02220a560f8c.png)
