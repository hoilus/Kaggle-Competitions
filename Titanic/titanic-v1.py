# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:15:19 2017

Titanic competitions on Kaggle
version 1 following the tutorial on dataquest

@author: hoilus
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation

"""-----------------------------------------
reading and looking at data
-----------------------------------------"""
# reading data
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
# print the first 5 rows of the dataframe.
print(titanic.head(5))
print(titanic_test.head(5))
# describe the dataset
print(titanic.describe())
print(titanic_test.describe())

"""-----------------------------------------
missing data
-----------------------------------------"""
# filling the missing data in 'Age' column by mean value of the original 'Age' feature
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

"""-----------------------------------------
converting the 'Sex' column
-----------------------------------------"""
# check the unique elements in 'Sex' column, expected to contain only male and female.
print(titanic["Sex"].unique())
# replace all the occurence of male and female with the number 0 and 1, respectively.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
print(titanic["Sex"].unique())

"""-----------------------------------------
converting the 'Embarked' column
-----------------------------------------"""
# check the unique elements in 'Embarked' column.
print(titanic["Embarked"].unique())
# fill the missing data using S
titanic["Embarked"] = titanic["Embarked"].fillna("S")
# replace Embarked values with numbers.
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
print(titanic["Embarked"].unique())

"""-----------------------------------------
Processing the test data
-----------------------------------------"""
# filling the missing data in 'Age' column by mean value of the original 'Age' feature
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
# replace all the occurence of male and female with the number 0 and 1, respectively.
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
# fill the missing data using S
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
# replace Embarked values with numbers.
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

"""-----------------------------------------
Logistic regression
-----------------------------------------"""
# The columns we wil use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
print(titanic[predictors].head(5))
print(titanic_test[predictors].head(5))
# Initiate our algorithm
alg = LogisticRegression(random_state = 1)
# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])
# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
