#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:06:38 2022

@author: thomasbaycroft
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import corner
from astropy import constants
import seaborn as sns
pi = np.pi
Ms = constants.M_sun.value
c = constants.c.value
G = constants.G.value
Rs = constants.R_sun.value
Mj = constants.M_jup.value
AU = constants.au.value
Me = constants.M_earth.value
from . import functions as fcs
from . import posteriors_file_analysis as pfa
from . import peak_finder as pf
from .RVdata_analysis import RV_planets, orbit
from . import make_plots as mplt
from . import orbit_plot as op
from . import make_datasheets as md


time0 = time.time()

def make_posterior_file(loc,epoch,binary,SB2,star_mass=1.0):
    #import the posterior samples as a dataframe and add a column which is the likelihood of each posterior sample
    with open(loc+'posterior_sample.txt') as f:
        names = f.readline().split()[1:]
    posterior_file = pd.read_csv(loc+'posterior_sample.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=names)
    
    posterior_info = pd.read_csv(loc+'posterior_sample_info.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=['Level','logL','tiebreak','ID'])
    posterior_file['logL'] = posterior_info['logL']
    
    
    maxNp, numpl, Bayes_factors, Nps = calc_nump_variables(posterior_file)
   
    ##############################################################################
    #from posterior sample file get the values/truth of various things:
    ##############################################################################
    
    #number of known objects
    nKOs = 0 
    testvalue = 0
    i=0
    while testvalue == 0:
        if 'KO_P'+str(i) in posterior_file.columns:
            nKOs +=1 
            i+=1
        else:
            testvalue = 1
    #is the precession of a binary orbit
    Precession = False
    if 'KO_wdot0' in posterior_file.columns:
        Precession = True
    
    ##############################################################################
    #add columns with other parameters, if this file exists: open it; if not: create it
    #then get maxlisolution and combine the posterior file to have a single planet column
    ##############################################################################
    
    try:
        posterior_file = pd.read_csv(loc+'posterior_file.csv')
    except FileNotFoundError:
        posterior_file= pfa.add_columns(posterior_file, epoch, binary, Precession, SB2, nKOs, maxNp,star_mass=star_mass)
        # a,b=[],[]
        # for i in range(len(posterior_file)):
        #     x,y = op.transit_pred(posterior_file[['M_pri','M_sec','KO_P0','KO_ecc0','Tper_bin','KO_w0','KO_wdot0']].values[i], posterior_file[['M_pl_0','P0','ecc0','Tper0','w0']].values[i], 2458800,epoch)
        #     a.append(x)
        #     b.append(y)
        # posterior_file['transit_pri'],posterior_file['transit_sec'] =a,b
        posterior_file.to_csv(loc+'posterior_file.csv',index=False)
        
def read_posterior_file(loc,res_numpl=False,nump=0.1,combine=False,columns=['P','K','phi','ecc','w','M_pl_','a_pl_','Tper']):
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    if res_numpl:
        maxNp, numpl, Bayes_factors, Nps = calc_nump_variables(posterior_file)
        if nump != 0.1:
            numpl = nump
        posterior_file = posterior_file[posterior_file['Np']==numpl] 
        if combine:
            columns_of_planetary_interest = columns
            posterior_file = pfa.combine_planets(posterior_file, maxNp, numpl, columns_of_planetary_interest)
        
    return posterior_file
        
def calc_nump_variables(posterior_file):
    #calculate the Bayes' Factor for each number of planets relative to one less 
    #and therefore the number of planets that we are going to plot
    maxNp = int(posterior_file['maxNp'][0])
    
    Bayes_factors = np.zeros(maxNp)
    Nps = np.zeros(maxNp+1)
    numpl = 0
    for i in range(maxNp):
        Nps[i] = max(np.sum([posterior_file['Np']==float(i)]),1)
        Bayes_factors[i] = max(np.sum([posterior_file['Np']==float(i+1)]),1)/max(np.sum([posterior_file['Np']==float(i)]),1)
        if Bayes_factors[i] >=10:
            numpl = i+1
    i+=1
    Nps[i] = max(np.sum([posterior_file['Np']==float(i)]),1)
    
    return maxNp, numpl, Bayes_factors, Nps

def get_MaxLi_samp(posts):
    return posts[posts['logL']==max(posts['logL'])].head(1)
    
    
 
def make_datasheets(loc, datafiles, instruments,SB2=False,binary=False):
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, Bayes_factors, Nps = calc_nump_variables(posterior_file)
    #now only consider the posteriors with the correct number of planets
    posteriors = posterior_file[posterior_file['Np']==numpl]
    
    #Get the maximum likelihood solution
    posts = posteriors
    #first without any cuts, the true maximum likelihood solution
    MaxLiSolution_nocut = posts[posts['logL']==max(posts['logL'])].head(1)
    
    #then if ecc not distinct from zero in the posteriors, choose the maximum likelihood with ecc<0.1
    for i in range(numpl):
        e_hist = np.histogram(posteriors['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
        if np.argmax(e_hist) in [0,1,2]:
            emax = 0.1
        else:
            emax = 1.0
        posts = posts[posts['ecc'+str(i)]<=emax]
    
    MaxLiSolution = posts[posts['logL']==max(posts['logL'])].head(1)
    
    
    
    #combine the columns for multiple planets into one for periodogram search
    columns_of_planetary_interest = ['P','K','phi','ecc','w','wdot','M_pl_','a_pl_','Tper']
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    
    ##############################################################################
    #find peaks in compined periodogram 
    ##############################################################################
    #currently using simple peak finding, could do with refining maybe more variables
    var = 'P'
    maxlivar=[]
    for i in range(numpl):
        maxlivar.append(MaxLiSolution[var+str(i)])
    indices = pf.simple_peaks(posteriors_single, var, maxlivar, numpl)
    
    ##############################################################################
    
    planet_params = MaxLiSolution[['P0','K0','Tper0','ecc0','w0','wdot0']]
    binary_params = MaxLiSolution[['KO_P0','KO_K0','Tper_bin','KO_ecc0','KO_w0','KO_wdot0']]
    
    for i, datafile in enumerate(datafiles):

        rvdata = pd.read_csv(datafile,sep='\t',engine='python',skiprows=[1],header=0)
        if SB2:
            rvdata_pri = rvdata[['bjd','vrad1','svrad1']]
            rvdata_sec = rvdata[['bjd','vrad2','svrad2']]
            rvdata_sec.columns,rvdata_pri.columns = ['bjd','vrad','svrad'],['bjd','vrad','svrad']
        else:
            rvdata_pri = rvdata[['bjd','vrad','svrad']]
    
    
        primary = RV_planets(rvdata_pri,'sophie_primary')
        if binary:
            primary.add_binary(binary_params,GR=True,precession=True)
    # primary.add_planet(1, planet_params)
    primary.add_vsys(MaxLiSolution['vsys'].values[0])
    primary.add_jitter(MaxLiSolution['extra_sigma'].values[0])
    
    secondary = RV_planets(rvdata_sec,'sophie_secondary')
    secondary.add_binary(binary_params,GR=True,precession=True,sb2=True,q=MaxLiSolution['KO_q0'])
    # secondary.add_planet(1, planet_params)
    secondary.add_vsys(MaxLiSolution['vsys_sec'].values[0])
    secondary.add_jitter(MaxLiSolution['extra_sigma_sec'].values[0])
    
    mplt.plot_rv_curves([primary],[secondary],colours=['forestgreen','darkorange'],inflate_errors=True)
    # tot_params = MaxLiSolution[['KO_P0','M_pri','M_sec','KO_ecc0','KO_w0']].values[0]
    # op.plot_circumbinary(*tot_params, [MaxLiSolution['P0']], [MaxLiSolution['M_pl_0']], [MaxLiSolution['ecc0']], [MaxLiSolution['w0']],primary.bjds,[MaxLiSolution['Tper_bin'],MaxLiSolution['Tper_bin'],MaxLiSolution['Tper0']])
    # op.plot_circumbinary(*tot_params, [float(MaxLiSolution['P0'])], [float(MaxLiSolution['M_pl_0'])], [float(MaxLiSolution['ecc0'])], [float(MaxLiSolution['w0'])],primary.bjds,[float(MaxLiSolution['Tper_bin']),float(MaxLiSolution['Tper_bin']),float(MaxLiSolution['Tper0'])])
    
    # fig, ax = plt.subplots(1)
    # a,b = mplt.make_detection_limit(posteriors_single,50,KM='K')
    # md.make_binary_sheet(posteriors_single, numpl,[Bayes_factors,Nps],[primary],[secondary],SB2=True)
    # bin_params = MaxLiSolution[['M_pri','M_sec','KO_P0','KO_ecc0','Tper_bin','KO_w0','KO_wdot0']].values[0]
    # pl_params = MaxLiSolution[['M_pl_0','P0','ecc0','Tper0','w0']].values[0]
    # start_time = 2458800
    # a,b = op.transit_pred(bin_params, pl_params, start_time,epoch)
    # print(a,b)
    # print(time.time() - time0)
    # pri = 2458883.390879
    # sec = 2458888.309427


# star_mass = 1.0
# #Tic172
# SB2 = True
# binary=True
# epoch = 2459514.6629
# datafile = 'TIC172_Dec2022.rdb'
# #HD78418
# SB2 = True
# binary=True
# epoch = 2459274.9281
# datafile = 'HD78418_rvs.rdb'
#HD195987
# SB2 = True
# binary=True
# epoch = 2459298.359
# epoch = 2457000
# # datafile = 'HD195987_rvs.rdb'
# loc = ''

def write_detlim(loc,KM='K'):
    
    if KM =='K':
        K_M = KM+'_ms'
    else:
        K_M = KM+'_M_sun'
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, Bayes_factors, Nps = calc_nump_variables(posterior_file)
    #now only consider the posteriors with the correct number of planets
    posteriors = posterior_file[posterior_file['Np']==numpl]
    
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_']
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    
    xax, detlim = mplt.make_detection_limit(posteriors_single,50,KM=KM)
    
    df = pd.DataFrame(np.array([xax,detlim]).T,columns=['Period','detlim_'+KM])
    df.to_csv(loc+'detlim_'+K_M+'.csv',index=False)
    

def get_percentile(x,y,bins,percentile=99,log=False):
    
    percs=[]
    bincentres=[]
    for i in range(len(bins)-1):
        y_bin = y[np.logical_and(x>=bins[i],x<bins[i+1])]
        percs.append(np.percentile(y_bin,percentile))
        if log:
            bincentres.append(10**((np.log10(bins[i])+np.log10(bins[i+1]))/2))
        else:
            bincentres.append((bins[i]+bins[i+1])/2)
            
    return bincentres, percs
