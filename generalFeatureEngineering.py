#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:19:24 2016

@author: anog
"""
from pandas import DataFrame
import numpy as np
from tools import splitDateTime

# Takes a column of dates (as in original data) and returns numeric time info as well as day of week
def returnDateDF(dateCol) : 
    l = dateCol.shape[0]
    cols = ["TIMESTAMP","YEAR","MONTH","DAY","HOUR"]
    res = DataFrame(np.nan, index = range(l), columns=cols)
    
    res["TIMESTAMP"] = dateCol.apply(lambda x:splitDateTime(x)[0])
    
    for i in range(1,len(cols)):
        feat = cols[i]
        res[feat] = dateCol.apply(lambda x:splitDateTime(x)[i+1])
    return res
    
# Checks if a group in a grouped feature corresponds to x. If it does, return said feature
def lastWeekFeat(x,groupedFeature):
    if x in groupedFeature.index : 
        return groupedFeature["CSPL_RECEIVED_CALLS"][x]
    else:
        return 0

# Returns vector of a given feature as measured last week
def returnLastWeekFeatureVect(groupedFeature,UID7Col,newname):
    return UID7Col.apply(lastWeekFeat,groupedFeature=groupedFeature).reindex(UID7Col.index).rename(newname)
    
def meanOnTimeWindow(timeStamps,response,windowSize):
    minTS = timeStamps.min()
    maxTS = timeStamps.max()
    
    limits= np.arange(minTS,maxTS,windowSize)
    res= DataFrame(np.nan, index=range(len(limits)),colums=["LIMINF","MEAN"])
    n = len(limits)
    
    for i in range(n-1):
        inf = limits[i]
        sup=limits[i+1]
        window = response[timeStamps>=inf and timeStamps<sup]
        res.ix[i] = (inf,window.mean())