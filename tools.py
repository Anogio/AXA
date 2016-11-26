#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:31:39 2016

@author: anog
"""
from datetime import datetime
from pandas import read_csv, concat

import math

import numpy as np

# Second count for a seven-day interval
sevenDays = 604800
# Total number of training lines
trainLines = 10878471
# Timestamp from which to try doing predictions, when doing cross validation
minTestDate = 1356649200

#Takes a date/time string as in the data and returns the information it contains
def splitDateTime(dateTime):
    spl = dateTime.split()
    date=spl[0]
    time=spl[1]
    
    spl = date.split("-")
    year =int( spl[0])
    month=int(spl[1])
    day = int(spl[2])
    
    spl=time.split(":")
    hour=int( spl[0])
    minute=int(spl[1])
    second=int( spl[2].split(".")[0] ) 
    
    dtm = datetime(year,month,day,hour,minute,second)
    timestamp = dtm.timestamp()
    weekd = dtm.weekday()
    days=["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    weekd = days[weekd]    

    return timestamp, weekd, year, month, day, hour, minute, second
    
# Used to import the train data from the .csv file
def importTrainData(nrows=None, otherCols=False):
    if otherCols:
        cols=["DATE","ASS_ASSIGNMENT","CSPL_RECEIVED_CALLS"]
    else:
        cols = ["DATE","WEEK_END","DAY_WE_DS","TPER_TEAM","ASS_ASSIGNMENT","ASS_DIRECTORSHIP","CSPL_RECEIVED_CALLS"]
    toCat = [ "DAY_WE_DS","TPER_TEAM", "ASS_ASSIGNMENT", "ASS_DIRECTORSHIP"]
    chunks = read_csv("train_2011_2012_2013.csv",usecols= cols, chunksize=100000, sep=";")
    frac = nrows/trainLines
    dat = concat([chunk.sample(frac = frac) for chunk in chunks] )
    for col in toCat:
        dat[col]=dat[col].astype("category")
        
    dat=dat.reset_index().drop("index",axis=1)
    return dat

# Used to import the train data from the submission file
def importTestData():
    dat= read_csv("submission.txt",sep="\t")
    dat["ASS_ASSIGNMENT"] = dat["ASS_ASSIGNMENT"].astype("category")
    return dat
        
# Builds a dictionnary recording what directorship each service is assigned to
#This function does not require the whole data, only the two relevant columns
def getHierarchy(dat):
    services = dat["ASS_ASSIGNMENT"].cat.categories
    hierarchy={}
    for service in services:
        ind = np.where(dat["ASS_ASSIGNMENT"]==service)[0][0]
        hierarchy[service]= dat["ASS_DIRECTORSHIP"][ind]
    return hierarchy
  
# From a date/time string, returns a boolean stating whether it is on a weekend
def isWE(dateTime):
    day = splitDateTime(dateTime)[1]
    return day == "Samedi" or day =="Dimanche"

# From a date/time string, returns "Jours" ou "Nuit" (the same values as in test
# data) depending on the time of day
def getShift(dateTime) :
    spl = splitDateTime(dateTime)
    hour = spl[5]
    minute = spl[6]

    if hour == 7:
        if minute>=30:
            return "Jours"
        else:
            return "Nuit"
    elif hour ==23:
        if minute < 30 : 
            return "Jours"
        else:
            return "Nuit"
    elif hour>7 and hour<23:
        return "Jours"
    else:
        return "Nuit"

# Assigns a unique ID to a given task, identified by the time interval and the 
# service it is assigned to. This will be used to identify which train data corresponds
# to an identical task seven days before
def uniqueID(timestamp,assignment):
    return hash(str(timestamp)+assignment)
    
def pandasLinExEval(y_true,y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    print(y_true.max())
    print(y_pred.max())
    alpha=0.1
    return ((alpha*(y_true-y_pred)).apply(math.exp) - alpha*(y_true-y_pred) -1).sum()
    
def linEx(true,pred):
    alpha=0.1
    return math.exp(alpha*(true-pred)) - alpha * (true - pred) -1

def selectNthComp(series,n):
    return series.apply(lambda x : x[n])