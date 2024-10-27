#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:55:01 2023

@author: thomasbaycroft
"""

import numpy as np
from scipy.stats import norm, t
from . import functions as fcs
import pandas as pd

class GAIAdata:
    '''
    Class which is given Gaia epoch astrometry data and can then manipulate it based on keplerian models
    '''
    
    def __init__(self,GAIAdata,t_ref=57388.5):
        
        self.data = GAIAdata
        self.times = self.data['bjd']
        self.w = self.data['w']
        self.w_err = self.data['sigw']
        self.psi = self.data['psi']
        self.pf = self.data['pf']
        
        self.t_ref = t_ref
        
        self.numpl = 0
        self.nKOs = 0
        self.num_bodies = 0
        self.planet_orbits = []
        self.KO_orbits = []
        self.jitter = 0.
        self.num_params = 0
        self.studT = False
        self.inliers = np.ones(len(self.times))
        self.outliers = np.zeros(len(self.times))
        
    def add_astrometric_solution(self,da,dd,mua,mud,parallax):
        
        self.wss = fcs.wss(da,dd,parallax,mua,mud,np.array(self.times),np.array(self.psi),np.array(self.pf),self.t_ref)
        
        # self.wss = (da + mua*(self.times-self.t_ref))*np.sin(self.psi) + (dd + mud*(self.times-self.t_ref))*np.cos(self.psi) + parallax*self.pf
        
    def add_jitter(self,jitter):
        self.jitter = jitter
        
    def add_planet(self, numpl, sample,Tpers=False):
        
        self.numpl = numpl
        
        
        for i in range(numpl):
            if Tpers:
                sample['Tper'+str(i)] = self.t_ref - float(sample['P'+str(i)])*float(sample['phi'+str(i)])/2/np.pi
            # print(sample[['P'+str(i),'a0'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i),'cosi'+str(i),'W'+str(i)]].values.tolist()[0])
            self.planet_orbits.append(orbit(sample[['P'+str(i),'a0'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i),'cosi'+str(i),'W'+str(i)]].values.tolist()[0]))
    
    def residuals(self):
        
        
        res = self.w - self.wss
        # print(res)
        for planet in self.planet_orbits:
            res -= planet.wk(self.psi,self.times)
            # print(res)
        return res
    
    def logL(self):
        
        # var = np.array(self.w_err)**2 + self.jitter**2
        logL = np.sum(np.log(norm(loc=0, scale=np.hypot(self.w_err, self.jitter)).pdf(self.residuals())))
        
        return logL

class orbit:
    
    def __init__(self,params,M0=1.,R0=1.,M1=1.,t_ref=57388.5):
        

        self.P = params[0]
        self.a0 = params[1]
        self.ecc = params[3]
        self.Tper = params[2]
        self.w =params[4]
        self.cosi = params[5]
        self.W = params[6]
        
        self.M0 = M0
        
        # self.Tper = t_ref - self.P*self.phi/2/np.pi
        
    def wk(self, psi, t):
        
        wk = fcs.wk_orb(self.P, self.Tper, self.ecc, self.cosi, self.W, self.w, self.a0, t, psi)
        
        return wk

    