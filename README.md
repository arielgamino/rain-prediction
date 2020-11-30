# Weather Department of Australia - Prediction Model

This is a three part implementation

## Installation

Use pipenv to install required modules

```bash
pip install pipenv
pipenv shell
```

## Project One
All notebooks implementing Project One are in the [part_one](part_one) folder

1. Exploratory Analysis to Understand the Weather Data.
2. Univariate and Bivariate Analysis - Understanding the Target Variable
3. Analyzing Missing Values and Outliers in Categorical and Numerical Features.
4. Data Preparation - Feature Engineering and Scaling

## Project Two
All notebooks implementing Project Two are in the [part_two](part_two) folder

1. Compute the accuracy of the model to understand how well the model
performs, to point out if the model is overfitting/underfitting, to compare it
to a dummy model and hence also understand the limitations of accuracy
as an evaluation metric leading to the need for more metrics.
2.  Compute the confusion matrix and classification report to understand the
various other metrics used in a classification problem and to judge
whether the values of the metrics are justified for the problem at hand.
3. Plot the ROC curves of the model and compare it to the ROC curves of
two extreme models, a random model, and an ideal model. Also compute
another common metric, the AUC, which is derived from the ROC plot.
4. To use and understand the importance of K-Fold Cross Validation when
evaluating a model. To also cross-validation in GridSearchCV to tune the
hyperparameters of the model to obtain the best possible model.


## Part Three
Deployment of Flask app locally and Heroku.

Module for cleaning data and prediction functions can be found in [model_package](model_package)
Flask app can be found in [predict_flask.py](predict_flask.py). 
The script that calls it for testing is this: [read_weather_predict.py](read_weather_predict.py)