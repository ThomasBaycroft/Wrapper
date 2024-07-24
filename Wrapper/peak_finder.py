#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:26:53 2023

@author: thomasbaycroft
"""
import numpy as np
# import hdbscan


def simple_peaks(posteriors,var,maxlivars,numpl,sensitivity=4,pre_index=None):
    '''
    Takes inputs:
        posteriors: dataframe of the posteriors in which we are looking for peaks
        var: str; the variable we are finding the peak in
        maxlivars: list of floats; the values for var for each planet in the maximum likelihood solution
        numpl: number of planets
        sensitivity: a scaling factor for sensitivity to peaks, if higher then we go wider around 
        peaks and allow shallower ones, but risk blurring them
    this function forms a histogram of the variable, identifies bins that are above a certain 
    threshold and groups them into 'peaks' it then identifies which peak is associated with each 
    solution and returns:
        indices: list of lists, for each planet there is a list of booleans indicating if each 
        posterior sample in 'posteriors' has a planet belonging to the corresponding 'peak'
    '''
    
    posts = posteriors
    indices = []
    for i in range(numpl):
        if pre_index != None:
            posteriors = posts[pre_index[i]]
        # maxliind = np.argmax(posteriors['logL'])
        # maxlivar = posteriors[var][maxliind]
        maxlivar = float(maxlivars[i])
        
        lower,upper = min(posteriors[var]),max(posteriors[var])
        
        number=50
        hist = np.histogram(posteriors[var],bins=np.logspace(np.log10(min(posteriors[var])),np.log10(max(posteriors[var])),number))
        isinpeak = np.r_[hist[0]>=-1+np.mean(hist[0])/sensitivity,np.array([0])]
        if sum(isinpeak)>=number*2/3:
            print(str(var)+' peak too wide, decreasing sensitivity')
            hist = np.histogram(posteriors[var],bins=np.logspace(np.log10(min(posteriors[var])),np.log10(max(posteriors[var])),number))
            isinpeak = np.r_[hist[0]>=-1+np.mean(hist[0]),np.array([0])]
            
        inpeak,lowers,uppers = False,[],[]
        for j,val in enumerate(isinpeak):
            if val==1 and inpeak==False:
                lowers.append(hist[1][j])
                inpeak = True
            elif val==0 and inpeak:
                uppers.append(hist[1][j])
                inpeak=False
        for j in range(len(lowers)):
            if lowers[j]<=maxlivar<=uppers[j]:
                lower, upper = lowers[j],uppers[j]   
        
        ind = np.logical_and(posteriors[var]<=upper,posteriors[var]>=lower)
        if pre_index == None:
            indices.append(ind)
        else:
            indices.append(np.logical_and(ind,pre_index[i]))
    
    return indices

def HDBScan(posteriors,Vars,maxlisamp,numpl):
    '''
    Takes inputs:
        posteriors: dataframe of the posteriors in which we are looking for peaks
        Vars: list of str; the variables we are clustering in
        maxlisamp: dataframe row; maximum likelihood solution
        numpl: number of planets

    This function performs clustering using HDBScan in the parameter space given and then associates clusters found with 
    the maximum likelohood solutions to obtain clusters associated with the detected planets and returns:
        indices: list of lists, for each planet there is a list of booleans indicating if each 
        posterior 
    '''
    pass
    
    
    