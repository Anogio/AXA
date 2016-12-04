#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:44:13 2016

@author: anog
"""
from tools import sevenDays, minTestDate, pandasLinExEval, selectNthComp, elapsedTimeString, positivePred
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import math
from sklearn.base import clone
from time import time, gmtime, strftime

#DEPRECATED: Predict on one train row
def classifEval1row(testRow,trainData,trainResponse,clf):
    ts = testRow["TIMESTAMP"]
    testRow = testRow.reshape(1, -1) 
    print('Select rows %d' % time())
    print(trainData.shape)
    userows = trainData["TIMESTAMP"]<=ts-sevenDays
    print("Fit %d" % time())
    print(trainData.loc[userows].shape)
    clf.fit(trainData.loc[userows],trainResponse[userows])
    
    pred = clf.predict(testRow)
    
    return pred
    
#Predict on a group of train rows. Used for the grid search step
def classifEvalGroup(testGroup,trainData,trainResponse,clf,tInit,i):
    t0=time()
    ts = testGroup["TIMESTAMP"].min()
    userows = trainData["TIMESTAMP"]<=ts-sevenDays

    print(trainData.loc[userows].shape)
    clf.fit(trainData.loc[userows],trainResponse[userows])
    
    pred = clf.predict(testGroup)
    
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    print("One group evaluated in :%s" % tstr)
    
    if i%10==0 and i>0 : 
        print("Time since beginning: %s" % elapsedTimeString(tInit))
    
    return pred
    
# Predict on a group of test rows. used for the prediction step
def makePredictionGroup(testGroup,trainData, trainResponse, clf) :
    t0=time()
    ts = testGroup["TIMESTAMP"].min()
    userows = trainData["TIMESTAMP"] <= ts - sevenDays

    clf.fit(trainData.loc[userows],trainResponse[userows])
    
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    print("One group evaluated in :%s" % tstr)
    
    return clf.predict(testGroup)
    
# Group test data then make full prediction
def makePrediction(trainData, trainResponse, testData, clf,groupByTS=False) :
    t0=time()
    if groupByTS:
        groups = testData.groupby("TIMESTAMP")
    else:
        groups = testData.groupby(["YEAR","MONTH","DAY"])
    print("Predicting on %d groups" % len(groups))
    pred = [pd.DataFrame(makePredictionGroup(g[1],trainData,trainResponse,clf)).set_index(g[1].index.values) for g in groups]
    print("Prediction Done. Concatenating...")
    pred = pd.concat(pred).sort_index()[0]
    print("Concatenation done")
    
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    print("Evaluation step completed in :%s" % tstr)
    
    print("Predicted values:")
    print(pred.describe())
    return pred
    
# Evaluate classifier through row by row prediction
def looClfEval(trainData,trainResponse, clfPlusDescr,frac = 1, lbd=1,rs=0):
    
    clf = clfPlusDescr[0]
    descr = str(clfPlusDescr[1])
    
    userows= trainData["TIMESTAMP"]>=minTestDate 
    t0=time()
    print("Evaluation of classifier for parameters %s" % descr)
    print("Predicting on lines:")
    print(trainData.loc[userows].sample(frac=frac,random_state=rs).shape)
    pred = trainData.loc[userows].sample(frac=frac,random_state=rs).apply(classifEval1row,axis=1,trainData=trainData,trainResponse=trainResponse,clf=clf).astype(float)
    err = pandasLinExEval(trainResponse.loc[userows],pred)
    totalErr = err.sum()
    multErr, multErrPen, expErr = testTransforms(trainResponse.loc[userows],pred,lbd)
    
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    print("Evaluation step completed in :%s" % tstr)

    
    print("Evaluation error : %d" % totalErr)
    print(err.describe())
    print("Expected values:")
    print(str(trainResponse.loc[userows].describe()))
    print("Predicted values:")
    print(pred.describe())
    print("________________________________________________________________")

    return totalErr, multErr, multErrPen, expErr

# Evaluate classifier by predicting on data grouped by day
def groupedClfEval(trainData,trainResponse, clfPlusDescr,t0,frac = 0.1, lbd=1,rs=1):
    clf = clfPlusDescr[0]
    descr = str(clfPlusDescr[1])
    userows= trainData["TIMESTAMP"]>=minTestDate
    t0=time()
    print("Evaluation of classifier for parameters %s" % descr)
    print("Predicting on lines:")
    print(trainData.loc[userows].sample(frac=frac,random_state=rs).shape)
    groups = trainData.loc[userows].groupby(["YEAR","MONTH","DAY"])
    print("Predicting on %d groups" % len(groups))
    inds = np.random.choice(len(groups),size= math.floor(len(groups)*frac),replace=False)
    print("Selected %d groups to predict on" % len(inds)) 

    pred =pd.concat( [pd.DataFrame(classifEvalGroup(g[1],trainData,trainResponse,clf,t0,i)).set_index(g[1].index.values) for (i,g) in enumerate(groups) if i in inds]).sort_index()
    pred=pred[0]    
    print(pred.shape)
    totalErr = pandasLinExEval(trainResponse.ix[pred.index.values],pred)

    multErr, multErrPen, expErr = testTransforms(trainResponse.ix[pred.index.values],positivePred(pred),lbd)
    
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    print("Evaluation step completed in :%s" % tstr)
    
    print("Evaluation error : %d" % totalErr)
    print("Expected values:")
    print(trainResponse.loc[userows].describe())
    print("Predicted values:")
    print(pred.describe())
    print("________________________________________________________________")

    return totalErr, multErr, multErrPen, expErr
    
# Test multiple classifiers to determine which one yields best predictions
# Also determine the best transformation on the prediction to improve precision
def customGridSearchCV(trainData, trainResponse, clf, argDict,tInit,frac=1,lbd=1,rs=0, method="noparall"):
    dicList = unravelDict(argDict)
    clfList = [(clone(clf),dic) for dic in dicList]    
    for i in range(len(clfList)) : 
        dic = dicList[i]
        clfList[i][0].set_params(**dic)
        
    t0=time()
    print("Evaluating classifier for %d fits" % len(clfList))
    num_cores = multiprocessing.cpu_count()
    if method == "noparall":
        results = [groupedClfEval(trainData,trainResponse,cl,tInit,frac,lbd,rs) for cl in clfList]
    elif method == "loo":
        results = Parallel(n_jobs=num_cores)(delayed(looClfEval)(trainData,trainResponse,cl,frac,lbd,rs) for cl in clfList)    
    else:
        results = Parallel(n_jobs=num_cores)(delayed(groupedClfEval)(trainData,trainResponse,cl,tInit,frac,lbd,rs) for cl in clfList)
        
    delta=gmtime(time() - t0)
    tstr=strftime('%H:%M:%S',delta)
    results= pd.Series(results)
    print("Full grid search completed in :%s" % tstr)    
    resBase = selectNthComp(results,0)
    resMult = selectNthComp(results,1)
    resMultPen = selectNthComp(results,2)
    resExp = selectNthComp(results,3)
    resMultCoeff = selectNthComp(resMult,0)
    resMultScore = selectNthComp(resMult,1)
    resMultPenCoeff = selectNthComp(resMultPen,0)
    resMultPenScore = selectNthComp(resMultPen,1)
    resExpCoeff = selectNthComp(resExp,0)
    resExpScore = selectNthComp(resExp,1)
    
    minIndBase = resBase.argmin()
    minIndMult = resMultScore.argmin()
    minIndMultPen = resMultPenScore.argmin()
    minIndExp = resExpScore.argmin()
    minCoeffMult =resMultCoeff[minIndMult]
    minCoeffMultPen = resMultPenCoeff[minIndMultPen]
    minCoeffExp = resExpCoeff[minIndExp]
    
 
    bestArgsBase = dicList[minIndBase]
    bestArgsMult = dicList[minIndMult]
    bestArgsMultPen = dicList[minIndMultPen]
    bestArgsExp = dicList[minIndExp]

    print("Best parameters for classifier:")
    print(bestArgsBase)
    print("Best parameters with multiplier")
    print(bestArgsMult)
    print("Best multiplier")
    print(minCoeffMult)
    print("Best parameters with multiplier and penalized coefficients")
    print(bestArgsMultPen)
    print("Best multiplier with penalization")
    print(minCoeffMultPen)    
    print("Best parameters with multiplier and penalized coefficients")
    print(bestArgsExp)
    print("Best multiplier with exponential correction")
    print(minCoeffExp)
    
    bestClfBase = clone(clf).set_params(**bestArgsBase)
    bestClfMult = clone(clf).set_params(**bestArgsMult)
    bestClfMultPen = clone(clf).set_params(**bestArgsMultPen)    
    bestClfExp = clone(clf).set_params(**bestArgsExp)
    return bestClfBase , (bestClfMult,minCoeffMult), (bestClfMultPen,minCoeffMultPen) , (bestClfExp, minCoeffExp)


def unravelDictAux(dictionary, acc):
    if dictionary =={} :
        return acc
    key,valueList = dictionary.popitem()
    temp = []
    for dic in acc :
        for val in valueList : 
            d= dic.copy()
            d[key] = val
            temp.append(d)
            
    return unravelDictAux(dictionary,temp)
    ()
# Creates list of all possible conbinations of keys in a dicitonary of lists
def unravelDict(dictionary):
    return unravelDictAux(dictionary.copy(),[{}])
    
# Apply multiplicative and exponential transforms to the data to try to make it match the shape of the output better
def testTransforms(true,pred,lbd = 1) : 
    ran = np.arange(0,20,0.1)
    res = pd.DataFrame(np.nan, index = range(len(ran)), columns = ["Multm","Err","ErrPen"])
    i=0
    for mult in ran:
            pred2 = mult * pred
            err = pandasLinExEval(true,pred2)
            errPen = err *( lbd*np.log10(mult)**2 + 1)
            res.ix[i] = [mult,err,errPen]
            i+=1
    m = res["Err"].argmin()
    mPen = res["ErrPen"].argmin()
    resm = res.ix[m]
    resmPen = res.ix[mPen]
    
    ran = np.arange(0,1,0.01)
    res = pd.DataFrame(np.nan, index = range(len(ran)), columns = ["Expm","Err"])
    
    for ex in ran:
        try:
            pred2 = (ex*pred).apply(math.exp)
            err = pandasLinExEval(true,pred2)
        except:
            err = np.inf
        res.ix[i] = [ex,err]        
        
    exp = res["Err"].argmin()
    resexp = res.ix[exp]
    

    
    return (resm["Multm"], resm["Err"]) , (resmPen["Multm"],resmPen["ErrPen"]) , (resexp["Expm"],resexp["Err"])
    
    
    
                
    
