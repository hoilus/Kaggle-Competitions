# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:36:53 2017

@author: hoilus

After self-learning two end-to-end machine-learning projects, 
I start to work on this version to attend Kaggle's competition.

The submission score is 0.76555

updated: engineering features
1. build a new feature based on family group.
result = titanic_train['Name'].loc[(titanic_train['Name'].str.contains('Palsson'))]
result = titanic_test['Name'].loc[(titanic_test['Name'].str.contains('Palsson'))]
2. Pclass, Cabin and Fare (which indiciates economic status) has great correlation to Survived values
result = titanic_train['Survived'].loc[(titanic_train['Cabin'].str.contains('0'))], here '0' replaces 'NaN',
shows a high death ratio.
result = titanic_train['Pclass'].loc[(titanic_train['Cabin'].str.contains('0'))]
result.mean()=2.575
shows 'NaN' of Cabin values highly indicates lower Pclass.
3. Cabin-A,B,C,D,E belongs to Pclass 1, Cabin-F belongs to Pclass 2, 
and Cabin-G belongs to Pclass 3, and Cabin-NaN belongs to Pclass 2 or 3.
4. Age and Pclass
result = titanic_train['Survived'].loc[(abs(titanic_train['Age']-35)<5) & (titanic_train['Pclass']==1)]
result.mean()=0.77
result = titanic_train['Survived'].loc[(titanic_train['Age']<10) & (titanic_train['Pclass']==2)]
result.mean()=1
5. Feature "Embarked" does not directly connect Survived values. Feature "Cabin" or "Pclass" does.

"""
"""
-----------------------------------------------------
Load Libraries and Data
----------------------------------------------------
"""
# Load libraries
import sys
import scipy
import numpy
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
##################################################
# Load dataset
titanic_train = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")
# Data shape
print(titanic_train.shape)
print(titanic_test.shape)
# Screen the data
print(titanic_train.head(10))
print(titanic_test.head(10))
# Statistical summary
print(titanic_train.describe())
print(titanic_test.describe())
"""
-----------------------------------------------------
Processing the Data
Here I assume passengers' economic status (Pclass, Cabin and Fare) largely determines their survival chance. 
Later I wil validate my assumption.
----------------------------------------------------
"""
# Relation of feature "Pclass" and "Survived", clearly showing the higher of economic status, the high survival chances.
titanic_train.groupby("Survived").Pclass.hist(alpha=0.4)
result0 = titanic_train['Survived'].loc[(titanic_train['Pclass']==1)]
result1 = titanic_train['Survived'].loc[(titanic_train['Pclass']==2)]
result2 = titanic_train['Survived'].loc[(titanic_train['Pclass']==3)]
Suratio_Pcl1 = result0.mean()
Suratio_Pcl2 = result1.mean()
Suratio_Pcl3 = result2.mean()
print(Suratio_Pcl1, Suratio_Pcl2, Suratio_Pcl3)
#################################################
### Is there any redundant features among (Pclass, Cabin and Fare)? And even feature "Embarked"?
## A. feature "Embarked" cannot directly connect to the survival. 
# It just reflects the combined survival ratio of "Pclass" boarding at the specific location.
# Therefore, "Embarked" is not a key feature.
## B. feature relation of  "Pclass" and "Fare"
# It clearly shows Fare depends on Pclass.
# Thus, feature "Fare" can be considered as redudant feature to "Pclass" till now.
# But, it might be useful for defining a new feature combining more than two original features.
titanic_train.groupby("Pclass").Fare.hist(alpha=0.4)
## C. feature  relation of "Pclass" and "Cabin"
# Fill NaN in feature "Cabin", and relearn it.
print(titanic_train["Cabin"].unique())
print(titanic_test["Cabin"].unique())
# The Cabin includes 'NaN' and 'A' to 'G'
titanic_train["Cabin"] = titanic_train["Cabin"].fillna('Z')
titanic_test["Cabin"] = titanic_test["Cabin"].fillna('Z')
result0 = titanic_train['Survived'].loc[(titanic_train['Cabin'].str.contains('Z'))]
result1 = titanic_train['Pclass'].loc[(titanic_train['Cabin'].str.contains('Z'))]
print(result0.mean(), result1.mean())
result_PcalCabin=[]
result_PcalCabin.append(result1.mean())
result_SurCabin=[]
result_SurCabin.append(result0.mean())
# The low survival ratio and avarge Pclass value 2.575 
# shows 'NaN' of Cabin values highly indicates lower Pclass (2 or 3).
for CabinIni in range(ord('A'), ord('H')):
    result_tmp = titanic_train['Pclass'].loc[(titanic_train['Cabin'].str.contains(chr(CabinIni)))]
    result_tmp1 = titanic_train['Survived'].loc[(titanic_train['Cabin'].str.contains(chr(CabinIni)))]
    result_PcalCabin.append(result_tmp.mean())
    result_SurCabin.append(result_tmp1.mean())
print(result_PcalCabin)
# Thus Cabin-A,B,C,D,E belongs to Pclass 1, Cabin-F belongs to Pclass 2, 
# Cabin-G belongs to Pclass 3, and Cabin-NaN belongs to Pclass 2 or 3.
print(result_SurCabin)
# The Cabin feature, which not only shows direct connection to "Pclass" 
# but defines more detailed categories, is better than "Pclass". Because,
# it probably includes the intrinsic relation of survival and cabin floors.
# replace 'Cabin' values with numbers
titanic_train["Cabin"] = titanic_train["Cabin"].astype(str)
titanic_train.loc[titanic_train["Cabin"].str.contains('Z'), "Cabin"] = '7'
print(titanic_train["Cabin"].unique())
# titanic_train["Cabin"] = titanic_train["Cabin"].fillna('1')
titanic_train.loc[titanic_train["Cabin"].str.contains('A'), "Cabin"] = '0'
titanic_train.loc[titanic_train["Cabin"].str.contains('B'), "Cabin"] = '1'
titanic_train.loc[titanic_train["Cabin"].str.contains('C'), "Cabin"] = '2'
titanic_train.loc[titanic_train["Cabin"].str.contains('D'), "Cabin"] = '3'
titanic_train.loc[titanic_train["Cabin"].str.contains('E'), "Cabin"] = '4'
titanic_train.loc[titanic_train["Cabin"].str.contains('F'), "Cabin"] = '5'
titanic_train.loc[titanic_train["Cabin"].str.contains('G'), "Cabin"] = '6'
titanic_train.loc[titanic_train["Cabin"].str.contains('T'), "Cabin"] = '7'
titanic_train["Cabin"] = titanic_train["Cabin"].astype(float)
titanic_train.loc[(titanic_train["Cabin"]==7) & (titanic_train["Pclass"]==2), "Cabin"] = 8
titanic_train.loc[(titanic_train["Cabin"]==7) & (titanic_train["Pclass"]==1), "Cabin"] = 9
print(titanic_train["Cabin"].unique())
print(titanic_train["Cabin"].describe())
# replace 'Cabin' values in titanic_test
titanic_test["Cabin"] = titanic_test["Cabin"].astype(str)
titanic_test.loc[titanic_test["Cabin"].str.contains('Z'), "Cabin"] = '7'
print(titanic_test["Cabin"].unique())
# titanic_train["Cabin"] = titanic_train["Cabin"].fillna('1')
titanic_test.loc[titanic_test["Cabin"].str.contains('A'), "Cabin"] = '0'
titanic_test.loc[titanic_test["Cabin"].str.contains('B'), "Cabin"] = '1'
titanic_test.loc[titanic_test["Cabin"].str.contains('C'), "Cabin"] = '2'
titanic_test.loc[titanic_test["Cabin"].str.contains('D'), "Cabin"] = '3'
titanic_test.loc[titanic_test["Cabin"].str.contains('E'), "Cabin"] = '4'
titanic_test.loc[titanic_test["Cabin"].str.contains('F'), "Cabin"] = '5'
titanic_test.loc[titanic_test["Cabin"].str.contains('G'), "Cabin"] = '6'
titanic_test.loc[titanic_test["Cabin"].str.contains('T'), "Cabin"] = '7'
titanic_test["Cabin"] = titanic_test["Cabin"].astype(float)
titanic_test.loc[(titanic_test["Cabin"]==7) & (titanic_test["Pclass"]==2), "Cabin"] = 8
titanic_test.loc[(titanic_test["Cabin"]==7) & (titanic_test["Pclass"]==1), "Cabin"] = 9
print(titanic_test["Cabin"].unique())
print(titanic_test["Cabin"].describe())
# adding Cabin dummies
cabin_dummies = pandas.get_dummies(titanic_train['Cabin'],prefix='Cabin')
titanic_train = pandas.concat([titanic_train, cabin_dummies], axis=1)
titanic_train.drop('Cabin', axis=1, inplace=True)
cabin_dummies = pandas.get_dummies(titanic_test['Cabin'],prefix='Cabin')
titanic_test = pandas.concat([titanic_test, cabin_dummies], axis=1)
titanic_test.drop('Cabin', axis=1, inplace=True)
#################################################
# replace all the occurence of male and female with the number 0 and 1, respectively.
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1
print(titanic_train["Sex"].unique())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
print(titanic_test["Sex"].unique())
#################################################
# fill NaN in feature "Age"
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
# discrete "Age"
titanic_train.loc[titanic_train["Age"]<=15, "Age"] = 0
titanic_train.loc[abs(titanic_train["Age"]-22.5)<=7.5, "Age"] = 1
titanic_train.loc[abs(titanic_train["Age"]-37.5)<=7.5, "Age"] = 2
titanic_train.loc[abs(titanic_train["Age"]-52.5)<=7.5, "Age"] = 3
titanic_train.loc[titanic_train["Age"]>60, "Age"] = 4
# discrete "Age"
titanic_test.loc[titanic_test["Age"]<=15, "Age"] = 0
titanic_test.loc[abs(titanic_test["Age"]-22.5)<=7.5, "Age"] = 1
titanic_test.loc[abs(titanic_test["Age"]-37.5)<=7.5, "Age"] = 2
titanic_test.loc[abs(titanic_test["Age"]-52.5)<=7.5, "Age"] = 3
titanic_test.loc[titanic_test["Age"]>60, "Age"] = 4
#################################################
# fill NaN in feature "Fare" of titani_test dataset
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#################################################
# create feature "Family" 
titanic_train["Family"] = titanic_train["SibSp"]+titanic_train["Parch"]
titanic_test["Family"] = titanic_test["SibSp"]+titanic_test["Parch"]
#################################################
# Extract titles from passenger names, 
# credits: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# titles reflect social status and may predict survival probability
titanic_train['Title'] = titanic_train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
titanic_test['Title'] = titanic_test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
titanic_train['Title'] = titanic_train.Title.map(Title_Dictionary)
titanic_test['Title'] = titanic_test.Title.map(Title_Dictionary)    
# drop 'Name' and adding title dummies
titanic_train.drop('Name', axis=1, inplace=True)
title_dummies = pandas.get_dummies(titanic_train['Title'],prefix='Title')
titanic_train = pandas.concat([titanic_train, title_dummies], axis=1)
titanic_train.drop('Title', axis=1, inplace=True)
titanic_test.drop('Name', axis=1, inplace=True)
title_dummies = pandas.get_dummies(titanic_test['Title'],prefix='Title')
titanic_test = pandas.concat([titanic_test, title_dummies], axis=1)
titanic_test.drop('Title', axis=1, inplace=True)    
"""
-----------------------------------------------------
Data Visualization
----------------------------------------------------
"""
# change data type of "Sex" and "Embarked" for data analysis and plot
# predictors = ["Pclass", "Sex", "Age", "Cabin", "Family", "Fare", "Title"]
predictors = list(titanic_train.columns)
predictors.remove('PassengerId')
predictors.remove('Survived')
predictors.remove('Ticket')
predictors.remove('Embarked')
predictors.remove('SibSp')
predictors.remove('Parch')
titanic_train[predictors] = titanic_train[predictors].astype(float)
titanic_test[predictors] = titanic_test[predictors].astype(float)
titanic_train[predictors] = titanic_train[predictors].apply(lambda x: x/x.max(), axis=0)
titanic_test[predictors] = titanic_test[predictors].apply(lambda x: x/x.max(), axis=0)
print(titanic_train[predictors].describe())
print(titanic_test[predictors].describe())
# box and whisker plots
# titanic_train.plot(kind='box', figsize=[10,10], subplots=True, layout=(6,2), sharex=False, sharey=False)
# plt.show()
pandas.options.display.mpl_style = 'default'
titanic_train.boxplot()
# histograms
titanic_train[predictors + ["Survived"]].hist(figsize=[10,10])
# Feature-'Survived' relationships
titanic_train.groupby("Survived").hist(figsize=[10,10])
titanic_train.groupby("Survived").Age.hist(alpha=0.4)
# scatter plot matrix, exploring feature-feature relationship
# define colors list, to be used to plot survived either red (=0) or green (=1)
# colors = ['red', 'green']
# scatter_matrix(titanic_train[predictors], figsize=[10,10], marker='x', c=titanic_train["Survived"].apply(lambda x:colors[x]))
scatter_matrix(titanic_train[predictors], alpha=0.2, figsize=[10,10], diagonal='kde')
"""
-----------------------------------------------------
Feature  Selection
----------------------------------------------------
"""
# Feature selection is  a process where you automatically select those features in your data 
# that contribute most to the prediction variable or output in which you are interested.
# Three benefits of performing feature selection before modeling the data:
# Reduces overfitting, improves accuracy, and readuces training time.
###############################################################
# A. Feature Importance
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
from sklearn.ensemble import ExtraTreesClassifier
array_FS = titanic_train[predictors+["Survived"]].values
X_FS = array_FS[:,0:21]
Y_FS = array_FS[:,21]
model = ExtraTreesClassifier()
model.fit(X_FS, Y_FS)
print(predictors)
print(model.feature_importances_)
###############################################################
# B. Univariate Selection
# Select those features that have the strongest relationship with the output variable.
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X_FS, Y_FS)
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_FS)
print(features[0:5, :])
###############################################################
# C. Recursive Feature Elimination
# The test below uses RFE with the logistic regression algorithm to select the top 3 features.
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X_FS, Y_FS)
print('Num Features: %d' % fit.n_features_)
print('Selected Features: %s' % fit.support_)
print('Feature Ranking: %s' % fit.ranking_)
###############################################################
# D. Principal Component Analysis
# PCA uses linear algebra to transform the dataset into a compressed form.
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fit = pca.fit(X_FS)
print('Explained  variance: %s' % fit.explained_variance_ratio_)
print(fit.components_)
##############################################################
##############################################################
"""
-----------------------------------------------------
Evaluate Predictive Algorithms
----------------------------------------------------
"""
##############################################################
# Create a validation dataset
# Split-out validation dataset
TRpredictors = predictors
array = titanic_train[TRpredictors+["Survived"]].values
X = array[:,0:21]
Y = array[:,21]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test Harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Build models
# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
"""
-----------------------------------------------------
Make Predictions
----------------------------------------------------
"""
# Make predictions on validation dataset
predsvc = GradientBoostingClassifier()
predsvc.fit(X_train, Y_train)
predictions = predsvc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""
-----------------------------------------------------
Evaluate Predictive Algorithms and make predictions using Random Forest
----------------------------------------------------
credits: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
"""
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
array_FS = titanic_train[predictors+["Survived"]].values
X_FS = array_FS[:,0:21]
Y_FS = array_FS[:,21]
model = ExtraTreesClassifier()
model.fit(X_FS, Y_FS)
features = pandas.DataFrame()
features['feature'] = predictors
features['importance'] = model.feature_importances_
print(predictors)
print(model.feature_importances_)
features.sort(['importance'],ascending=False)

model_tr = SelectFromModel(model,prefit=True)
train_new = model_tr.transform(titanic_train[predictors])
train_new.shape
test_new = model_tr.transform(titanic_test[predictors])
test_new.shape

forest = RandomForestClassifier()
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,220,230,240,250,260,270,280,290],
                 'criterion': ['gini','entropy']
                 }
cross_validation = StratifiedKFold(Y_FS, n_folds=10)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_new, Y_FS)
#grid_search.fit(X_FS, Y_FS)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
"""
-----------------------------------------------------
Submit Predictions
----------------------------------------------------
"""
# Make predictions on validation dataset
gbc = GradientBoostingClassifier()
gbc.fit(X, Y)
predictions = gbc.predict(titanic_test[TRpredictors])
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission-03142017', spe='\t', encoding='utf-8')
