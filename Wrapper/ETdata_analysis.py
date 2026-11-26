#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:52:59 2023

@author: thomasbaycroft
"""

import numpy as np
from . import functions as fcs
import pandas as pd

class ET_planets():
    '''
    Class which is given Eclipse timing (or transit timing?) data and can then manipulate it based on ephemeris and keplerian LTTE models.
    '''
    
    def __init__(self,ETdata,ref_T_type='first'):
        
        self.ETdata = ETdata
        self.Epoch = self.ETdata['Epoch']
        self.T0 = self.ETdata['T0']
        self.sigT0 = self.ETdata['sigT0']
        if ref_T_type=='first':
            self.ref_T = self.T0[0]
        self.num_bodies = 0
        self.planet_orbits = []
        
        
    def add_ephemeris(self,ephem1,ephem2=0,ephem3=0):
        
        self.ephem1 = float(np.asarray(ephem1).item())
        self.ephem2 = float(np.asarray(ephem2).item())
        self.ephem3 = float(np.asarray(ephem3).item())
    
    def add_keplerians(self, numpl,params):
        
        self.numpl = numpl
        self.num_bodies += numpl
        
        for i in range(numpl):
            self.planet_orbits.append(et_orbit(params[['P'+str(i),'K'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i)]].values.tolist()[0],self.ephem1,self.ref_T))
            
    def remove_ephems_from_data(self,values):
        
        return values- fcs.ephem_etv(self.Epoch, self.ref_T, self.ephem1,self.ephem2,self.ephem3)
    
    def remove_orbit_from_data(self,orbit,values=[],epochs=[]):
        
        if len(epochs)==0:
            epochs = self.Epoch
        if len(values) == 0:
            values = self.T0
        
        return values - orbit.ET_Keplerian(epochs)
    
    def add_orbit_to_data(self,orbit,values=[],epochs=[]):
        
        if len(epochs)==0:
            epochs = self.Epoch
        if len(values) == 0:
            values = self.T0
        
        return values + orbit.ET_Keplerian(epochs)
    
    def residuals(self):
        
        res = self.remove_ephems_from_data(self.T0)
        for orbit in self.planet_orbits:
            res = self.remove_orbit_from_data(orbit,values=res)
            
        return res
    
    def LTTE_orbit(self,orbit):
        
        return self.add_orbit_to_data(orbit,self.residuals(),self.Epoch)
    
    def rms(self):
        resids = np.array(self.residuals())
        return (sum(resids**2)/len(resids))**(1/2)
            

class et_orbit():
    
    def __init__(self,params,ephem1,ref_T):
        
        self.P = params[0]
        self.K = params[1]
        self.ecc = params[3]
        self.Tper = params[2]
        self.w =params[4]
        
        self.ephem1 = ephem1
        self.ref_T = ref_T
        
        
    def ET_Keplerian(self, epochs):
        
        return fcs.keplerian_etv(epochs, self.ref_T, self.ephem1, self.P, self.K, self.ecc, self.w, self.Tper)
    
    def phase(self, epochs):
        
        times=epochs*self.ephem1
        
        return fcs.phase(times, self.P, self.Tper)