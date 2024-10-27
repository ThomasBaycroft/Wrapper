#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 22:17:06 2023

@author: thomasbaycroft
"""
import numpy as np
from scipy.stats import norm, t
from . import functions as fcs
import pandas as pd

class RV_planets:
    '''
    Class which is given radial velocity data and can then manipulate it based on keplerian models
    '''
    
    def __init__(self,RVdatas,inst_name,bis_and_FWHM=False,RV_units='kms'):
        '''
        Takes:
            RVdatas: dataframe
                column 0: times (bjd)
                column 1: vrad (km/s or m/s)
                column 2: svrad (same units as vrad)
            inst_name: str; name of the instrument/name associated with this dataset
            bis_and_FWHM: bool; are columns 3 and 4 of RVdatas the bisector_span and the fwhm indicators
            RV_units: what units the rv data are in, defaults to km/s 
        '''
        
        self.RVdatas = RVdatas
        self.bjds = self.RVdatas['bjd']
        self.vrad = self.RVdatas['vrad']
        self.svrad = self.RVdatas['svrad']
        self.inst_name = inst_name
        self.bisfw = bis_and_FWHM
        if self.bisfw:
            self.bis = self.RVdatas['bis_span']
            self.fwhm = self.RVdatas['fwhm']
        self.binary = False
        self.numpl = 0
        self.nKOs = 0
        self.num_bodies = 0
        self.drift_order = 0
        self.planet_orbits = []
        self.KO_orbits = []
        self.rvunits = RV_units
        self.vsys = 0.
        #convert units to m/s
        if self.rvunits == 'kms':
            self.vrad *= 1000
            self.svrad *= 1000
            if self.bisfw:
                self.bis *= 1000
        self.jitter = 0.
        self.num_params = 0
        self.studT = False
        self.inliers = np.ones(len(self.bjds))
        self.outliers = np.zeros(len(self.bjds))
        self.Bin_precession = False
        
    def add_binary(self,params,precession=False,GR=False,Tides=False,M0=1.,M1=1.,R0=1.,sb2=False,q=1.):
        '''
        Add a binary star, given:
            params: list of keplerian parameters [Period, Semi-amplitude, time of pericentre passage,
                                            eccentricity, argument of pericentre, rate of apsidal precession]
            precession,GR,Tides: bools; flags as to whether precession, GR correction and Tidal correction are applied
            M0,M1,R0: maasses and radii of relevant stars, used if tidal and/or GR corrections used (solar units)
            sb2: bool; is this the data on the secondary of an sb2
            q: if so then this should be the mass ratio
        '''
        
        self.binary = True
        self.num_bodies += 1
        self.num_params += 5 + precession
        parameters = params[:]
        if type(params) == pd.core.frame.DataFrame:
            parameters = params.values.tolist()[0]
        
        if precession==False:
            parameters[5] = 0.
        self.Bin_precession=precession   
        if sb2:
            parameters[4] -= np.pi
            parameters[1] *= 1/q
            
        self.binary_orbit = orbit(parameters,GR=GR,Tides=Tides,M0=M0,M1=M1,R0=R0)
            
    def add_vsys(self,vsys,units='ms'):
        scaling=1.0
        if units == 'kms':
            scaling=1000
        self.vsys = float(vsys*scaling)
        self.num_params+=1
        
    def add_offset(self,offset,units='ms'):
        scaling=1.0
        if units == 'kms':
            scaling=1000
        print(self.inst_name,float(offset),float(offset)*scaling,self.vrad[0])
        self.vrad -= float(offset)*scaling
        self.num_params+=1
        
    def add_jitter(self,jitter):
        self.jitter = float(jitter)
        self.num_params+=1
        
        
    def add_planet(self,numpl,params,precession=False):
        '''
        Add a planet, given:
            numpl: int; number of planets being added
            params: list of keplerian parameters: 
                [Period, Semi-amplitude, time of pericentre passage,eccentricity, argument of pericentre, rate of apsidal precession]
                there should be one of each of these for each planet added named/numbered as below
            precession: boos; flags as to whether precession is included
        '''
        
        self.numpl = numpl
        self.num_bodies += numpl
        
        self.num_params+= (5+precession)*numpl
        
        
        if precession==False:
            for i in range(numpl):
                params['wdot'+str(i)] = 0.
        
        for i in range(numpl):
            self.planet_orbits.append(orbit(params[['P'+str(i),'K'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i),'wdot'+str(i)]].values.tolist()[0]))
            
    def add_known_obj(self,nKOs,params,precession=False,start_ind=1):
        '''
        Add a planet, given:
            nKOs: int; number of known objects being added total (including the above binary which will not be added again)
            params: list of keplerian parameters: 
                [Period, Semi-amplitude, time of pericentre passage,eccentricity, argument of pericentre, rate of apsidal precession]
                there should be one of each of these for each planet added named/numbered as below
            precession: boos; flags as to whether precession is included
        '''
        
        self.nKOs = nKOs
        self.num_bodies += nKOs-1
        
        self.num_params+= (5+precession)*self.numpl
        
        if precession==False:
            for i in range(nKOs):
                params['KO_wdot'+str(i)] = 0.
        
        for i in range(start_ind,nKOs+start_ind):
            self.KO_orbits.append(orbit(params[['KO_P'+str(i),'KO_K'+str(i),'KO_Tper'+str(i),'KO_ecc'+str(i),'KO_w'+str(i),'KO_wdot'+str(i)]].values.tolist()[0]))
            
    def add_drift(self,order,params):
        
        self.drift_params = []
        self.drift_order = order
        var=['slope','quadr','cubic']
        for i in range(order):
            self.drift_params.append(float(params[var[i]]))
        for i in range(order,3):
            self.drift_params.append(0)
        
        self.remove_drifts()
        self.num_params+= order
        
    def remove_orbit_from_data(self,data_times,data_values,orbit):
                
        return data_values - orbit.RV_keplerian_sub(data_times)
    
    def add_orbit_to_data(self,data_times,data_values,orbit):
        
        return data_values + orbit.RV_keplerian_sub(data_times)
    
    def rvs_no_binary(self):
        
        if self.binary:
            rvs = self.remove_orbit_from_data(self.bjds,self.vrad,self.binary_orbit)
        else:
            rvs = self.vrad
            
        return rvs
    
    def remove_drifts(self):
        
        tmid = (max(self.bjds)+min(self.bjds))/2
        
        drift = self.drift_params[0]*(self.bjds-tmid) +self.drift_params[1]*(self.bjds-tmid)**2 + self.drift_params[2]*(self.bjds-tmid)**3
        self.vrad -= drift
        
    def drift(self,times='data'):
        
        if times == 'data':
            times = self.bjds
        elif times == 'cont':
            times = np.arange(min(self.bjds),max(self.bjds),0.1)
        
        tmid = (max(self.bjds)+min(self.bjds))/2
        
        drift = self.drift_params[0]*(times-tmid) +self.drift_params[1]*(times-tmid)**2 + self.drift_params[2]*(times-tmid)**3
        
        return drift
        
    def residuals(self):
        
        res = self.vrad
        if self.binary:
            res = self.remove_orbit_from_data(self.bjds,res,self.binary_orbit)
        for planet in self.planet_orbits:
            res = self.remove_orbit_from_data(self.bjds, res, planet)
        for KO in self.KO_orbits:
            res = self.remove_orbit_from_data(self.bjds, res, KO)
            
        res = res-self.vsys
        
        return res
    
    def rv_curve_orbit(self,orbit):
        
        return self.add_orbit_to_data(self.bjds,self.residuals(),orbit)
    
    def rv_curve_binary(self):
        
        return self.rv_curve_orbit(self.binary_orbit)
    
    def rv_curve_planet(self,number=0):
        
        return self.rv_curve_orbit(self.planet_orbits[number])
    
    def calc_stud_T_outliers(self,nu,sensitivity=10):
        
        Gd = norm(loc=0, scale=np.hypot(self.svrad, self.jitter)).pdf(self.residuals())
        Td = t(df=nu, loc=0, scale=np.hypot(self.svrad, self.jitter)).pdf(self.residuals())
        self.studT_ratios = Td/Gd
        for j in range(len(Td)):
            if Td[j] == 0. and Gd[j] == 0.:
                self.studT_ratios[j] = 1.
        self.inliers = self.studT_ratios <= sensitivity
        self.outliers = self.studT_ratios > sensitivity
        self.studT=True
        self.nu = nu
        
    
    def rms(self):
        resids = np.array(self.residuals())
        return (sum(resids**2)/len(resids))**(1/2)
    
    def logL(self):
        var = np.array(self.svrad)**2 + self.jitter**2
        if self.studT:
            logL = np.sum(np.log(t(df=self.nu, loc=0, scale=np.hypot(self.svrad, self.jitter)).pdf(self.residuals())))
        else:
            # logL = np.sum(- 0.5*np.log(2*np.pi) - 0.5*np.log(var) - 0.5*np.array(self.residuals())**2/var)
            logL = np.sum(np.log(norm(loc=0, scale=np.hypot(self.svrad, self.jitter)).pdf(self.residuals())))
        return logL
      
    def mean_rv_err(self):
        return sum(self.svrad)/len(self.svrad)
        
    def chi_2_nu(self):
        #*****#
        nobs = len(self.bjds)
        fit_params = self.num_params
        nu = nobs-fit_params
        
        return (np.array(self.residuals)**2)/(np.array(self.svrad)**2),nu
    
    def write_reses(self,loc,system,prisec=''):
        columns=['bjd','vrad']
        file=[self.bjds,self.vrad]
        if self.binary:
            columns.append('vrad-bin')
            file.append(self.remove_orbit_from_data(self.bjds, self.vrad, self.binary_orbit))
        columns.append('residuals')
        file.append(self.residuals())
        columns.append('svrad')
        file.append(self.svrad)
        if self.studT:
            columns.append('T_ratio')
            file.append(self.studT_ratios)

        df = pd.DataFrame(np.array(file).T,columns=columns)
        df.to_csv(loc+system+self.inst_name+prisec+'_residuals.csv',index=False)
        
    
        
    
class orbit:
    
    def __init__(self,params,GR=False,Tides=False,M0=1.,R0=1.,M1=0):
        

        self.P = params[0]
        self.K = params[1]
        self.ecc = params[3]
        self.Tper = params[2]
        self.w =params[4]
        self.wdot = params[5]
        
        self.GR = GR
        self.Tides = Tides
        self.M0 = M0
        self.M1 = M1
        self.R0 = R0

    def mass(self):
        return fcs.get_mass(self.K,self.M0+self.M1,self.P,self.ecc)


    def RV_keplerian_sub(self,times):
        
        if self.GR or self.Tides:
            rvs = fcs.keplerian_plusplus(times, self.P, self.K, self.ecc, self.w, self.wdot,self.Tper, 0, self.M0, self.R0, self.M1, self.GR, self.Tides)
        else:
            rvs = fcs.keplerian_plus(times, self.P, self.K, self.ecc, self.w, self.wdot,self.Tper, 0)
        
        return rvs
    
    def RV_keplerian_add(self,times,phase=False):
        
        if phase:
            kep = fcs.keplerian_phase(times, self.P, self.K, self.ecc, self.w,self.Tper, 0)
        else:
            kep = fcs.keplerian(times, self.P, self.K, self.ecc, self.w,self.Tper, 0)
            
        return kep
        
    def phase(self,times):
        
        return fcs.phase(times, self.P, self.Tper)
    
    def true_anoms(self,times):
        
        phase = self.phase(times)
        M = phase*2*np.pi
        E = fcs.ecc_anomaly(M, self.ecc)
        nu = fcs.true_anomaly(E, self.ecc)
        
        
        return nu
    
    def true_anom_hist(self,times,bins=10,nu_tod=0):
        
        nu=self.true_anoms(times)

        less = np.array(nu<nu_tod)
        
        nu += less*2*np.pi
        
        hist = np.histogram(nu,bins)
        
        return hist
    
    def Tbins_in_trueanom(self,today,nbins,times):
        
        nu_tod = self.true_anoms(today)
        
        nubins = np.linspace(nu_tod,nu_tod+2*np.pi,nbins+1)
        
        nubins = nubins.T[0]
        
        Ebins = 2*np.arctan(np.sqrt((1-self.ecc)/(1+self.ecc))*np.tan(nubins/2))
        
        Mbins = Ebins - self.ecc*np.sin(Ebins)
        
        Tbins = Mbins*self.P/2/np.pi + self.Tper
                
        Tbins += today - Tbins[0]
        
        less = np.array(Tbins<=Tbins[0])
        
        less[0] = 0
        
        Tbins += less*self.P
        
        # nubins += less*2*np.pi

        hist = self.true_anom_hist(times,nubins,nu_tod=nu_tod)

        return hist,Tbins
        
        
        
        
        
        
    