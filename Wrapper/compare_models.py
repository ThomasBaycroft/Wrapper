#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:15:58 2023

@author: thomasbaycroft
"""

from . import run_analysis as ra
from . import RVdata_analysis as rda
from . import functions as fcs
from . import peak_finder as pf
from . import posteriors_file_analysis as pfa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# import random

colours = ['r','b','g','m','c','darkviolet','orange']
colours = ['indianred','cornflowerblue','g','m','c','darkviolet','orange']

class model_divergence:
    
    def __init__(self):
        
        self.number_of_models = 0
        self.models = []
        self.datatimes = []
        self.time0 = 0
        self.time1 = 0
        
    def add_data(self,datatimes):
        self.datatimes = np.array(datatimes)
        self.time0 = min(datatimes)
        self.time1 = max(datatimes)
        
    def add_model(self,orbits,model_type,likes):
        self.number_of_models+=1
        new_model = model_group(orbits,model_type,likes)
        self.models.append(new_model)
        
    def plot_best(self,extention_time):
        for i,model in enumerate(self.models):
            ind = model.maxli_ind
            times = np.arange(self.time0,self.time1+extention_time,0.1)
            plt.plot(times,model.kep_curve(times,ind),c=colours[i])
            plt.scatter(self.datatimes,model.kep_curve(self.datatimes,ind),c=colours[i])
        
    def plot_dist(self,extention_time,num):
        for i,model in enumerate(self.models):
            if len(model.models) < num:
                print('too many samples wanted/too few samples in the model')
            else:
                inds = np.random.randint(0,len(model.models),num)
                times = np.arange(self.time0,self.time1+extention_time,0.1)
                for ind in inds:
                    plt.plot(times,model.kep_curve(times,ind),c=colours[i],alpha=6/num)
                plt.scatter(self.datatimes,model.kep_curve(self.datatimes,model.maxli_ind),c=colours[i])
                
    def plot_diffs_best(self,extension_time,model1_num=0,model2_num=1):
        model1, model2 = self.models[model1_num], self.models[model2_num]
        times = np.arange(self.time0,self.time1+extension_time,0.1)
        ind1, ind2 = model1.maxli_ind, model2.maxli_ind
        plt.plot(times,model1.kep_curve(times,ind1)-model2.kep_curve(times,ind2),c='orange')
        plt.scatter(self.datatimes,model1.kep_curve(self.datatimes,ind1)-model2.kep_curve(self.datatimes,ind2),c='purple')
        plt.axhline(0,ls='--',c='black',alpha=0.7)
        
        
    def plot_diffs_dist(self,extension_time,num,model1_num=0,model2_num=1):
        model1, model2 = self.models[model1_num], self.models[model2_num]
        times = np.arange(self.time0,self.time1+extension_time,0.1)
        if len(model1.models) < num:
            print('too many samples wanted/too few samples in model 1')
        elif len(model2.models) < num:
            print('too many samples wanted/too few samples in model 2')
        else:
            inds1, inds2 = np.random.randint(0,len(model1.models),num), np.random.randint(0,len(model2.models),num)
            for i in range(num):
                plt.plot(times,model1.kep_curve(times,inds1[i]) - model2.kep_curve(times,inds2[i]),c='orange',alpha=6/num)
            plt.scatter(self.datatimes,model1.kep_curve(self.datatimes,model1.maxli_ind)-model2.kep_curve(self.datatimes,model2.maxli_ind),c='purple')
            plt.axhline(0,ls='--',c='black',alpha=0.7)

class model_group:
    
    def __init__(self,models,model_type, likelihoods):
        
        self.Likelihoods = likelihoods
        self.model_type = model_type
        self.models = models
        self.maxli_ind = np.argmax(self.Likelihoods)
        self.maxlimod = models[self.maxli_ind]
            
        
    def kep_curve(self, times,ind):
        if self.model_type == 'RV':
            return self.models[ind].RV_keplerians(times)

#make a model class that can account for a number of keplerians, maybe drifts too etc
#then model group can hold a whole bunch of these to then plot random ones
class model:
    
    def __init__(self):
        self.number_of_orbits = 0
        self.orbits = []
        
    def kima_sample_planets(self,post,numpl,tper=True,M0epoch=None):
        for i in range(numpl):
            if tper:
                params = post[['P'+str(i),'K'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i)]].values.tolist()
                params.append(0)
            else:
                params = post[['P'+str(i),'K'+str(i),'phi'+str(i),'ecc'+str(i),'w'+str(i)]].values.tolist()
                params[2] = M0epoch - params[0]*params[2]/2/np.pi
                params.append(0)
            orb = rda.orbit(params)
            self.add_orbit(orb)
        
    def add_orbit(self,orbit):
        self.orbits.append(orbit)
        self.number_of_orbits+=1
        
    def RV_keplerians(self,times):
        rvs = np.zeros_like(times)
        for orb in self.orbits:
            rvs += orb.RV_keplerian_add(times)
        return rvs
            