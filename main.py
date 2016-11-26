#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:33:07 2016

@author: anog
"""
import pandas as pd
import numpy as np

from tools import importTrainData, importTestData 
from trainFeatureEngineering import returnID, returnDayMinusSevenID
from testFeatureEngineering import makeDateFeatures, makeDirFeature, makeDayMinusSevenID
from generalFeatureEngineering import returnDateDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from classification import customGridSearchCV, makePrediction

nrows= 1000000

# Import data
print("Importing data")
train = importTrainData(nrows)
y= train["CSPL_RECEIVED_CALLS"]
train = train.drop("CSPL_RECEIVED_CALLS",axis=1)
test = importTestData()
pred = test["prediction"]
test=test.drop("prediction",axis = 1)
print(train.shape)
print(test.shape)
# Engineer train features
print("Making features for train dataset")
train["UID-7"] = returnDayMinusSevenID(train[["DATE","ASS_ASSIGNMENT"]])
print(train.shape)
# Engineer test features
print("Making features for test dataset")
test= makeDateFeatures(test)
test= makeDirFeature(test,train[["ASS_ASSIGNMENT","ASS_DIRECTORSHIP"]])
test = makeDayMinusSevenID(test)
print(test.shape)
# Reorder colums so that train and test have same structure
print("Reordering test columns")
test = test[train.columns.tolist()]

### At this point, train and test have exactly the same structure. This means that any transformation 
### made on of of the dataframe should also be applied to the other, so that a predictor can be 
### trained on the first one and used to predict on the other
print("Making features for both datasets")
train = pd.concat([train, returnDateDF(train["DATE"])], axis=1)
test  = pd.concat([test,  returnDateDF(test["DATE"]) ], axis=1)

train = train.drop("DATE",axis=1)
test  = test. drop("DATE",axis=1)
print(train.shape)
print(test.shape)
# Get features from seven days before
#trainPast = importTrainData(None, True)
#toAdd = train.loc[train["UID-7"].isin(test["UID-7"].tolist())]
#toAdd = toAdd.drop(["DATE","ASS_ASSIGNMENT"])

catCols = test.select_dtypes(include=["category"]).columns
test = pd.get_dummies(test,columns=catCols)
train = pd.get_dummies(train,columns=catCols)
print(train.shape)
print(test.shape)
toRemove = np.setdiff1d(train.columns,test.columns)
train.drop(toRemove, inplace= True, axis = 1)
print(train.shape)
print(test.shape)

clf = RandomForestRegressor()
param = {
             "n_estimators":[3],
             "max_features":["auto","sqrt"]
         }
print("Performing grid search")
CVResults = customGridSearchCV(train,y,clf,param,frac=0.1,rs=1,lbd=1)
bestClfBase = CVResults[0]
bestClfMult = CVResults[1][0]
MultCoeff = CVResults[1][1]
bestClfMultPen = CVResults[2][0]
MultPenCoeff = CVResults[2][1]

print("Predicting")
predBase = makePrediction(train,y,test,bestClfBase)
predMult = MultCoeff * makePrediction(train,y,test,bestClfMult)
predMultPen =MultPenCoeff *  makePrediction(train,y,test,bestClfMultPen)

print("Making final submission")
submission = importTestData()

submission["response"]=predBase
submission.to_csv("submission.txt",sep="\t")

submission["response"]=predBase
submission.to_csv("submissionMult.txt",sep="\t")

submission["response"]=predBase
submission.to_csv("submissionMultPen.txt",sep="\t")
print("Done")
