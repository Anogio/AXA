#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:33:07 2016

@author: anog
"""
import pandas as pd
import numpy as np
import gc

from tools import importTrainData, importTestData , truncatePred
from trainFeatureEngineering import returnID, returnDayMinusSevenID
from testFeatureEngineering import makeDateFeatures, makeDirFeature, makeDayMinusSevenID
from generalFeatureEngineering import returnDateDF, returnLastWeekFeatureVect
from sklearn.ensemble import RandomForestRegressor
from classification import customGridSearchCV, makePrediction
from xgboost import XGBRegressor

nrows= 3 * 10**6
buildUID = True
CVTrainFrac = 0.3

# Import data
print("Importing data")
train = importTrainData(nrows,quick=False)
y= train["CSPL_RECEIVED_CALLS"]
train.drop("CSPL_RECEIVED_CALLS",axis=1,inplace=True)
test = importTestData()
pred = test["prediction"]
test=test.drop("prediction",axis = 1)
print(train.shape)
print(test.shape)
# Engineer train features
print("Making features for train dataset")
if buildUID:
    train["UID-7"] = returnDayMinusSevenID(train[["DATE","ASS_ASSIGNMENT"]])

print(train.shape)
# Engineer test features
print("Making features for test dataset")
test= makeDateFeatures(test)
test= makeDirFeature(test,train[["ASS_ASSIGNMENT","ASS_DIRECTORSHIP"]])
if buildUID:
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

train.drop("DATE",axis=1,inplace=True)
test  = test. drop("DATE",axis=1)
print(train.shape)
print(test.shape)
# Get features from seven days before
if buildUID:
    train2 = importTrainData(None,otherCols=True,quick=True)
    train2["UID"] = returnID(train2[["DATE","ASS_ASSIGNMENT"]])
    y2 = train2["CSPL_RECEIVED_CALLS"]
    train2.drop("CSPL_RECEIVED_CALLS",axis=1,inplace=True)
    responseAssignment = pd.concat([train2["UID"],y2],axis=1)
    del train2
    del y2
    gc.collect()
    groups = responseAssignment.groupby("UID")
    maxPerDay = groups.max()
    maxPerDay.to_csv("maxGroupedPerUID.csv")

    train = pd.concat([train, returnLastWeekFeatureVect(maxPerDay,train["UID-7"],"MAX_LASTWEEK")], axis=1)
    test  = pd.concat([test,  returnLastWeekFeatureVect(maxPerDay,test["UID-7"],"MAX_LASTWEEK")], axis=1)
    
    meanPerDay=groups.mean()
    meanPerDay.to_csv("meanGroupedPerUID.csv")
    train = pd.concat([train, returnLastWeekFeatureVect(meanPerDay,train["UID-7"],"MEAN_LASTWEEK")], axis=1)
    test  = pd.concat([test,  returnLastWeekFeatureVect(meanPerDay,test["UID-7"],"MEAN_LASTWEEK")], axis=1)

    

catCols = test.select_dtypes(include=["category"]).columns
test = pd.get_dummies(test,columns=catCols)
train = pd.get_dummies(train,columns=catCols)
print(train.shape)
print(test.shape)
toRemove = np.setdiff1d(train.columns,test.columns)
train.drop(toRemove, inplace= True, axis = 1)
print(train.shape)
print(test.shape)

clf = XGBRegressor()
param = {
             "max_depth":[6],
             "gamma":[0,0.5],
             "subsample":[0.7],
             "colsample_bytree":[0.7],
             "reg_alpha":[0,0.5],
             "reg_lambda":[0]
         }

print("Performing grid search")
CVResults = customGridSearchCV(train,y,clf,param,frac=CVTrainFrac,rs=1,lbd=1.5)
bestClfBase = CVResults[0]
bestClfMult = CVResults[1][0]
MultCoeff = CVResults[1][1]
bestClfMultPen = CVResults[2][0]
MultPenCoeff = CVResults[2][1]

print("Making final submission")
submission = importTestData()


print("Predicting")
predBase = makePrediction(train,y,test,bestClfBase)
submission["prediction"]=predBase
submission.to_csv("newSubmission.txt",sep="\t")
submission["prediction"] = truncatePred(y,predBase)
submission.to_csv("newSubmissionTrunc.txt",sep="\t")
del predBase

print("Predicting with multiplier")
predMult = MultCoeff * makePrediction(train,y,test,bestClfMult)
submission["prediction"]=predMult
submission.to_csv("newSubmissionMult.txt",sep="\t")
submission["prediction"] = truncatePred(y,predMult)
submission.to_csv("newSubmissionTrunc.txt",sep="\t")
del predMult

print("Predicting with multiplier and penalization")
predMultPen =MultPenCoeff *  makePrediction(train,y,test,bestClfMultPen)
submission["prediction"]=predMultPen
submission.to_csv("newSubmissionMultPen.txt",sep="\t")
submission["prediction"] = truncatePred(y,predMultPen)
submission.to_csv("newSubmissionTrunc.txt",sep="\t")
print("Done")
