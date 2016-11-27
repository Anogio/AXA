#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:46:29 2016

@author: anog
"""
from tools import splitDateTime, isWE, getShift, getHierarchy, uniqueID, sevenDays, writeDict

#Adds to the test data some date features already contained in the train data
def makeDateFeatures(df) : 
    df["DAY_WE_DS"] = df["DATE"].apply(lambda x:splitDateTime(x)[1]).astype("category")
    df["WEEK_END"] = df["DATE"].apply(isWE).astype(int)
    df["TPER_TEAM"]=df["DATE"].apply(getShift).astype("category")
    return df
# Adds to the test data the data on which directorship the task was assigned to.
# #This function does not require the whole train data, only the two relevant columns
# However, as it directly adds the column to the test data, the whole dataframe must be given
def makeDirFeature(testData,trainData):
    hierarchy = getHierarchy(trainData)
    writeDict(hierarchy,"hierarchy.txt")
    testData["ASS_DIRECTORSHIP"]= testData["ASS_ASSIGNMENT"].apply(lambda x:hierarchy[x]).astype("category")
    testData["ASS_ASSIGNMENT"]=testData["ASS_ASSIGNMENT"].astype("category")
    
    return testData
    
# Builds a new feature that identifies the ID of the same task seven days before. 
# This will be used to 
def makeDayMinusSevenID(df):
    df["UID-7"] = df.apply(lambda row : uniqueID(splitDateTime(row["DATE"])[0] - sevenDays, row["ASS_ASSIGNMENT"]),axis=1)
    return df
