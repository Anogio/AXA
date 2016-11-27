#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:19:24 2016

@author: anog
"""
from pandas import DataFrame
import numpy as np
from tools import splitDateTime

def returnDateDF(dateCol) : 
    l = dateCol.shape[0]
    cols = ["TIMESTAMP","YEAR","MONTH","DAY","HOUR"]
    res = DataFrame(np.nan, index = range(l), columns=cols)
    
    res["TIMESTAMP"] = dateCol.apply(lambda x:splitDateTime(x)[0])
    
    for i in range(1,len(cols)):
        feat = cols[i]
        res[feat] = dateCol.apply(lambda x:splitDateTime(x)[i+1])
    return res
    
def lastWeekFeat(x,groupedFeature):
    if x in groupedFeature.index : 
        return groupedFeature["CSPL_RECEIVED_CALLS"][x]
    else:
        return 0

def returnLastWeekFeatureVect(groupedFeature,UID7Col,newname):
    return UID7Col.apply(lastWeekFeat,groupedFeature=groupedFeature).reindex(UID7Col.index).rename(newname)
