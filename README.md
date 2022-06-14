# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [File Description](#Description)
3. [Instructions](#Instructions)
4. [Results](#Results)


## Project Motivation <a name="motivation"></a>

In this project I was interested in exploring the Airbnb this information was obtained from the Seattle Airbnb Open Data Kaggle site and that is available for everyone who wants to consult it, . In this dataset we can found information about the availability, prices, charactheristics of the listed places. I wanted to find out what characteristics could make the price of a place publiWhat type of rooms has the highest proportion of reservations in the next 60 days?
shed out there greater or lesser, to try to find out this, I asked myself the following questions:

*  What time of year has the highest and lowest prices?
*  What type of rooms has the highest proportion of reservations in the next 60 days?
*  The number of bedrooms influences the price?
*  The number of bathrooms influences the price?
*  How is the relationship between property type price of them?



## Files Description <a name="Description"></a>

The files that were used in the project are the following, you can download them here https://www.kaggle.com/airbnb/seattle

- **calendar.csv**: Provides date, availability, and price data for each listing
- **listings.csv**: Provides comprehensive details about each listing including host details, location description and details, and review ratings
- **reviews.csv**: Provides details of review comments


## Instructions <a name="Instructions"></a>

Follow the steps below to run the app

1. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To change the location to the directory app write `cd app` and then `python run.py` to run the web app

3. Visit this link to see the Dashboard  http://0.0.0.0:3001/

## Results <a name="Results"></a>

The main findings of the code can be found at the ![image](https://user-images.githubusercontent.com/100497288/162784170-dd551273-df7c-476f-8937-02220a560f8c.png)

 




![image](https://user-images.githubusercontent.com/100497288/162784170-dd551273-df7c-476f-8937-02220a560f8c.png)
