# MultiClass-Classification-Model

## Motivation
When we think about maching learning, often times people pay our attention to building ML models. That is, how to achieve higher accuracy, which algorithm to use and how to go about feature engineering etc. However, how are we going to actually use these models is often neglected. And yet, this is equally as important, if not more, as building model and is crucial part of building end to end ML pipeline. Only when a model is fully integrated with business systems, we can produce real values from the models

## Project Overview
The main goal of this project is to build a model, write production code and create pipeline and serve the model via Flask API. As mentioned in the Motivation section, I am focusing on building the most accurate model but rather creating the entire ML pipeline from researching to deployment. The project also follows some of the standard software engineering practices such as logging, versioning and testing to make the process as painless and easy as possible in the future. 

## Model Overview
I use Stacking Ensemble method for the model in this project. For more info about it read [this](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) Also I only select top 10 features to make training faster and keep the model simple. In terms of improving the model accuray, I suggest trying rebalancing target variable, using more features, more robust preprocessing and tuning the model. 

## Setup and Requirements
run `pip install -r requirements.txt` to install <br /> 
run `python setup.py develop` to modularize files <br /> 
<br /> 
for this project requirements are as following:
#### For model building
* numpy>=1.18.1,<1.19.0
* pandas>=0.25.3,<0.26.0
* scikit-learn>=0.22.1,<0.23.0
* joblib>=0.14.1,<0.15.0
* lightgbm == 2.3.0

#### for testing
* pytest>=5.3.2,<6.0.0

#### for packaging
setuptools >= 41.4.0,<42.0.0


## Running and Testing 
I use tox to automate training and testing. Simply run `tox`. If one wishes to see the output of the model, run train and predict separately
For testing flask api, run `pytest test_controller.py`
