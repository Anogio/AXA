#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:00:02 2016

@author: anog
"""
from tools import uniqueID, splitDateTime, sevenDays

#This function returns the column of unique IDs. Howeer, it does not add the 
#vector to the dataframe and return it like for the test data. This is because
#the training dataframe is much larger that the test, so we want to be able to 
#input only the relevant columns rather than the whole data (of which a copy is
# created when the function is called)
def returnID(df):
    return df.apply(lambda row : uniqueID(splitDateTime(row["DATE"])[0], row["ASS_ASSIGNMENT"]), axis=1)
    
def returnDayMinusSevenID(df):
    return df.apply(lambda row : uniqueID(splitDateTime(row["DATE"])[0] - sevenDays, row["ASS_ASSIGNMENT"]), axis=1)
    
