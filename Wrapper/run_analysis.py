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
# import seaborn as sns
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

def make_posterior_file_RV(loc,epoch,binary,SB2,star_mass=1.0,star_mass_err=0.0):
    #import the posterior samples as a dataframe and add a column which is the likelihood of each posterior sample
    with open(loc+'posterior_sample.txt') as f:
        names = f.readline().split()[1:]
    posterior_file = pd.read_csv(loc+'posterior_sample.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=names)
    
    posterior_info = pd.read_csv(loc+'posterior_sample_info.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=['Level','logL','tiebreak','ID'])
    posterior_file['logL'] = posterior_info['logL']
    
    
    maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file)
   
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
        posterior_file= pfa.add_columns_RV(posterior_file, epoch, binary, Precession, SB2, nKOs, maxNp,star_mass=star_mass,star_mass_err=star_mass_err)
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
        maxNp, numpl,nKOs, Bayes_factors, Nps, BF_samps = calc_nump_variables(posterior_file)
        if nump != 0.1:
            numpl = nump
        posterior_file = posterior_file[posterior_file['Np']==numpl] 
        if combine:
            columns_of_planetary_interest = columns
            posterior_file = pfa.combine_planets(posterior_file, maxNp, numpl, columns_of_planetary_interest)
        
    return posterior_file

def Samples_BFs(loc,maxNp):
    
    
    weights = pd.read_csv(loc+'weights.txt',header=None)
    with open(loc+'sample.txt') as f:
        names = f.readline()[1:].split()
    samps = pd.read_csv(loc+'sample.txt',header=None,skiprows=[0],names=names,sep='\s+',engine='python',usecols=['Np'])['Np']
    
    wts = np.zeros(maxNp+1)
    for i in range(maxNp+1):
        locs = samps==i
        wts[i] = np.sum((weights.T*locs).T)
        
    BFs = wts[1:]/wts[:-1]
    
    return BFs

def FIP_TIP(loc,maxNp,P_prior,dataspan):
    
    Pmin, Pmax = P_prior[0], P_prior[1]
    # print(Pmin,Pmax)
    
    Likes_samps = np.array(pd.read_csv(loc+'sample_info.txt',header=None,skiprows=[0],sep='\s+',engine='python',usecols=[1]))
    weights = pd.read_csv(loc+'weights.txt',header=None)
    
    cols = []
    for i in range(maxNp):
        cols.append('P'+str(i))
        
    with open(loc+'sample.txt') as f:
        names = f.readline()[1:].split()
    P_samps = np.array(pd.read_csv(loc+'sample.txt',header=None,skiprows=[0],names=names,sep='\s+',engine='python',usecols=cols))
    P_posts = np.array(pd.read_csv(loc+'posterior_sample.txt',skiprows=[0,1],names=names,sep='\s+',engine='python',usecols=cols))
    
    logL = Likes_samps - np.max(Likes_samps)
    prob_samps = np.exp(logL)*weights
    prob_samps = prob_samps/np.sum(prob_samps)
    prob_posts = np.ones(len(P_posts))/len(P_posts)
    
    omegas_samps = 2*np.pi/P_samps
    omegas_posts = 2*np.pi/P_posts
    
    omegas = np.flip(np.linspace(2*np.pi/Pmax,2*np.pi/Pmin,10000))#can change nuumber of points
    
    nom = len(omegas)

    tiper_samps = np.zeros(nom)
    tiper_posts = np.zeros(nom)
    
    freq_int = 2*np.pi/dataspan*1 #frequency resolution
    
    for i in range(nom):
        #lower and upper bound
        bm = np.max([omegas[i] - freq_int/2,0])
    
        bM = omegas[i] + freq_int/2
        tM_samps = np.logical_and(omegas_samps>=bm, omegas_samps<bM)
        tM_posts = np.logical_and(omegas_posts>=bm, omegas_posts<bM)
    
        trutharr_samps = np.any(tM_samps, axis=1)
        trutharr_posts = np.any(tM_posts, axis=1)
    
        tiper_samps[i] = np.sum(prob_samps[trutharr_samps]) #TIP-periodogram
        tiper_posts[i] = np.sum(prob_posts[trutharr_posts]) #TIP-periodogram
    
    fiper_samps = 1-tiper_samps
    fiper_posts = 1-tiper_posts
    pers = 2*np.pi/omegas
    
    return pers, fiper_posts, fiper_samps
        
        
        
def calc_nump_variables(posterior_file,loc=None):
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
    
    Bayes_factors_samps = Bayes_factors
    if loc != None:
        Bayes_factors_samps = Samples_BFs(loc,maxNp)
        
    nKOs = 0
    found = False
    while found == False:
        try:
            P = posterior_file['KO_P'+str(nKOs+1)]
            nKOs+=1
        except:
            found = True
    
    return maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps

def get_MaxLi_samp(posts):
    return posts[posts['logL']==max(posts['logL'])].head(1)

def read_rv_data(file,SB2=False,bisFW=False,skiprows=[1]):
    
    rvdata = pd.read_csv(file,sep='\t',engine='python',skiprows=skiprows,header=0,comment='#')
    datas = []
    try:
        if SB2:
            rvdata_pri = rvdata[['bjd','vrad1','svrad1']]
            rvdata_sec = rvdata[['bjd','vrad2','svrad2']]
            rvdata_sec.columns,rvdata_pri.columns = ['bjd','vrad','svrad'],['bjd','vrad','svrad']
            datas.append(rvdata_pri)
            datas.append(rvdata_sec)
        else:
            if bisFW:
                rvdata_pri = rvdata[['bjd','vrad','svrad','bis_span','fwhm']]
            else:
                rvdata_pri = rvdata[['bjd','vrad','svrad']]
            datas.append(rvdata_pri)
    except KeyError:
        if SB2:
            rvdata_pri = rvdata[['jdb','vrad1','svrad1']]
            rvdata_sec = rvdata[['jdb','vrad2','svrad2']]
            rvdata_sec.columns,rvdata_pri.columns = ['bjd','vrad','svrad'],['bjd','vrad','svrad']
            datas.append(rvdata_pri)
            datas.append(rvdata_sec)
        else:
            if bisFW:
                rvdata_pri = rvdata[['jdb','vrad','svrad','bis_span','fwhm']]
                rvdata_pri.columns = ['bjd','vrad','svrad','bis_span','fwhm']
            else:
                rvdata_pri = rvdata[['jdb','vrad','svrad']]
                rvdata_pri.columns = ['bjd','vrad','svrad']
            datas.append(rvdata_pri)
    
    return datas
    
 
def make_datasheets(system,outputloc,inputloc,datafiles,instruments,bis_fw=False,SB2=False,binary=False,precession=False,GR=False,Tides=False,P_prior=[1,2000],CMR=[1.0,1.0],enfnp=0):
    
    loc = inputloc
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl,nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file,loc)
    if enfnp >=1:
        numpl = enfnp
    #now only consider the posteriors with the correct number of planets
    len_posts = len(posterior_file)
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
    
    if SB2:
        pass
    else:
        MaxLiSolution['M_pri'] = CMR[0]
        if binary:
            MaxLiSolution['R_pri'] = CMR[1]
            MaxLiSolution['M_sec'] = fcs.masses(SB2,float(MaxLiSolution['KO_K0']),float(MaxLiSolution['KO_P0']),float(MaxLiSolution['KO_ecc0']),q=1.0,M1=float(MaxLiSolution['M_pri']))[1]
    
    #combine the columns for multiple planets into one for periodogram search
    columns_of_planetary_interest = ['P','K','phi','ecc','w','wdot','M_pl_','a_pl_','Tper']
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_','Tper']
    #for posterior file restricted to numpl
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    #for the full posterior file (excluding not found planets)
    full_posteriors = pfa.combine_planets(posterior_file,maxNp,maxNp,columns_of_planetary_interest)
    full_posteriors = full_posteriors[full_posteriors['P']>0.0]
    
    
    ##############################################################################
    #find peaks in combined periodogram 
    ##############################################################################
    #currently using simple peak finding, could do with refining maybe more variables
    variables = ['P','K']
    variables_2 = ['KO_P','KO_K']
    indices=None
    indices_bin=None
    indices_full=None
    if binary:
        for var in variables_2:
            maxlivar=[]
            maxlivar.append(MaxLiSolution[var+str(0)])
            indices_bin = pf.simple_peaks(posteriors_single, var+str(0), maxlivar, 1, pre_index=indices_bin)
        indices_bin=indices_bin[0]    
        indices_bin = posteriors_single['KO_P0']!=0
    # for var in variables_2:
    #     maxlivar=[]
    #     for i in range(nKOs):
    #         maxlivar.append(MaxLiSolution[var+str(i)])
    #     indices = pf.simple_peaks(posteriors_single, var, maxlivar, numpl, pre_index=indices)
    #     indices_full = pf.simple_peaks(full_posteriors, var, maxlivar, numpl, pre_index=indices_full)
    for var in variables:
        maxlivar=[]
        for i in range(numpl):
            maxlivar.append(MaxLiSolution[var+str(i)])
        indices = pf.simple_peaks(posteriors_single, var, maxlivar, numpl, pre_index=indices)
        indices_full = pf.simple_peaks(full_posteriors, var, maxlivar, numpl, pre_index=indices_full)

    
    ##############################################################################
    
    if binary:
        binary_params = MaxLiSolution[['KO_P0','KO_K0','Tper_bin','KO_ecc0','KO_w0','KO_wdot0']]
    
    primaries = []
    secondaries = []
    numdats = []
    for i, datafile in enumerate(datafiles):

        # rvdata = pd.read_csv(datafile,sep='\t',engine='python',skiprows=[1],header=0)
        if SB2:
            rvdata_pri,rvdata_sec = read_rv_data(datafile,True)[0],read_rv_data(datafile,True)[1]
        #     rvdata_pri = rvdata[['bjd','vrad1','svrad1']]
        #     rvdata_sec = rvdata[['bjd','vrad2','svrad2']]
        #     rvdata_sec.columns,rvdata_pri.columns = ['bjd','vrad','svrad'],['bjd','vrad','svrad']
        else:
            rvdata_pri = read_rv_data(datafile,SB2=False,bisFW=bis_fw)[0]
        #     rvdata_pri = rvdata[['bjd','vrad','svrad']]
        
        
        primary = RV_planets(rvdata_pri,instruments[i],bis_and_FWHM=bis_fw)
        if len(datafiles)>=2 and i+1<len(datafiles):
            primary.add_offset(MaxLiSolution['offset'+str(i+1)])
        primary.add_planet(numpl,MaxLiSolution)
        primary.add_vsys(MaxLiSolution['vsys'])
        if len(datafiles)==1:
            primary.add_jitter(MaxLiSolution['extra_sigma'])
        else:
            primary.add_jitter(MaxLiSolution['jitter'+str(i+1)])
        primary.add_drift(0,MaxLiSolution)
        if binary:
            primary.add_binary(binary_params,GR=GR,Tides=Tides,precession=precession,M0=float(MaxLiSolution['M_pri']),M1=float(MaxLiSolution['M_sec']),R0=float(MaxLiSolution['R_pri']))
            if SB2:
                secondary = RV_planets(rvdata_sec,instruments[i])
                if i>=1:
                    secondary.add_offset(MaxLiSolution['offset_sec'+str(i)])
                secondary.add_binary(binary_params,GR=GR,Tides=Tides,precession=precession,sb2=True,q=float(MaxLiSolution.KO_q0),M0=float(MaxLiSolution['M_sec']),M1=float(MaxLiSolution['M_pri']))
                secondary.add_planet(numpl,MaxLiSolution)
                secondary.add_vsys(MaxLiSolution['vsys_sec'])
                if len(datafiles)==1:
                    secondary.add_jitter(MaxLiSolution['extra_sigma_sec'])
                else:
                    secondary.add_jitter(MaxLiSolution['jitter_sec'+str(i+1)])
                secondary.add_drift(0,MaxLiSolution)
                secondaries.append(secondary)
                secondary.write_reses(outputloc+'datafiles/', system,prisec='_sec')
                primary.write_reses(outputloc+'datafiles/', system,prisec='_pri')
            else:
                primary.write_reses(outputloc+'datafiles/', system)
        primaries.append(primary)

        
        
        numdats.append(len(primary.bjds))
    
    mints=[]
    maxts=[]
    for pri_data in primaries:
        mints.append(min(pri_data.bjds))
        maxts.append(max(pri_data.bjds))
    dataspan = float(max(maxts) - min(mints))
    
    #Make the various datasheets
    datasheet_num = 1
    md.make_binary_sheet(system,outputloc,posteriors_single, len_posts, indices, indices_bin, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=precession)
    datasheet_num += 1
    for i in range(nKOs):
        md.make_object_sheet(system,outputloc,True,i,datasheet_num,posteriors_single, indices, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=False)
        datasheet_num += 1
    for i in range(numpl):
        md.make_object_sheet(system,outputloc,False,i,datasheet_num,posteriors_single, indices, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=False)
        datasheet_num += 1
    md.make_timeseries_sheet(system,outputloc,datasheet_num,instruments, MaxLiSolution, numpl, nKOs, primaries,sec_data_list=secondaries,binary=binary,SB2=SB2)    
    datasheet_num +=1 
    fiptip = FIP_TIP(loc, maxNp, P_prior, dataspan)
    md.make_extras_sheet(system,outputloc,datasheet_num,full_posteriors, indices_full, MaxLiSolution,fiptip,binary=binary)
    datasheet_num+=1
    md.make_diagnostics(system, outputloc, datasheet_num, loc)
    
    # inds = np.ones(len(posteriors_single))
    if binary:
        inds = indices_bin
    elif numpl>=1:
        inds = indices[0]
    else:
        inds = np.zeros(len(posteriors_single))
    for ind in indices:
        inds = np.logical_or(inds,ind)
    
    # Make a table of parameters
    nul='---'
    param_columns = ['parameter','MaxLi_value','Median_value','+','-','units']
    params = []
    params.append(['number of data',sum(numdats),nul,nul,nul,nul])
    params.append(['number of planets',numpl,nul,nul,nul,nul])
    vsyss = posteriors_single['vsys'][inds]
    params.append(['vsys',np.round(float(MaxLiSolution['vsys']),8),np.round(np.percentile(vsyss,50),8),np.round(np.percentile(vsyss,84)-np.percentile(vsyss,50),8),np.round(np.percentile(vsyss,50)-np.percentile(vsyss,16),8),'m/s'])
    if SB2:
        vsys2s = posteriors_single['vsys_sec'][inds]
        params.append(['vsys_sec',np.round(float(MaxLiSolution['vsys_sec']),1),np.round(np.percentile(vsys2s,50),8),np.round(np.percentile(vsys2s,84)-np.percentile(vsys2s,50),8),np.round(np.percentile(vsys2s,50)-np.percentile(vsys2s,16),8),'m/s'])
    params.append(['reference epoch',round(float(MaxLiSolution['Epoch']),8),nul,nul,nul,'bjd'])
    params.append(['data timespan',round(dataspan),nul,nul,nul,'days'])
    for i,pri in enumerate(primaries):
        if i >=1:
            param = posteriors_single['offset'+str(i)][inds]
            params.append(['offset',np.round(float(MaxLiSolution['offset'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        if SB2:
            if i >=1:
                param = posteriors_single['offest_sec'+str(i)][inds]
                params.append(['offset_sec',np.round(float(MaxLiSolution['offset_sec'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
            sec = secondaries[i]
            merr2,rms2,jit2 = round(sec.mean_rv_err(),4),round(sec.rms(),4),round(sec.jitter,4)
        else:
            merr2,rms2,jit2 = nul,nul,nul
        params.append([nul,instruments[i],nul,nul,nul,nul])
        params.append(['mean RV_err',round(pri.mean_rv_err(),4),merr2,nul,nul,'m/s'])
        params.append(['RMS',round(pri.rms(),4),rms2,nul,nul,'m/s'])
        params.append(['Jitter',round(pri.jitter,4),jit2,nul,nul,'m/s'])
    for i in range(maxNp):
        params.append(['Bayes Factor '+str(i+1)+'/'+str(i),Bayes_factors[i],Bayes_factors_samps[i],nul,nul])
    if binary:
        params.append(['','Binary','','','',''])
        param = posteriors_single['M_pri'][inds]
        params.append(['M_pri',np.round(float(MaxLiSolution['M_pri']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['M_sec'][inds]
        params.append(['M_sec',np.round(float(MaxLiSolution['M_sec']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['KO_P0'][inds]
        params.append(['P_obs',np.round(float(MaxLiSolution['KO_P0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['a_pri'][inds]
        params.append(['a_pri',np.round(float(MaxLiSolution['a_pri']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['a_sec'][inds]
        params.append(['a_sec',np.round(float(MaxLiSolution['a_sec']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['KO_K0'][inds]
        params.append(['K',np.round(float(MaxLiSolution['KO_K0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['KO_ecc0'][inds]
        params.append(['e',np.round(float(MaxLiSolution['KO_ecc0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['KO_w0'][inds]
        params.append(['w',np.round(float(MaxLiSolution['KO_w0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_phi0'][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['KO_phi0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['Tper_bin'][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['Tper_bin']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])
        if primaries[0].Bin_precession == True:
            param = posteriors_single['KO_wdot0'][inds]
            params.append(['w_dot',np.round(float(MaxLiSolution['KO_wdot0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'arcsec/yr'])
            param = posteriors_single['KO_Pano0'][inds]
            params.append(['P_ano',np.round(float(MaxLiSolution['KO_Pano0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        if SB2:
            param = posteriors_single['KO_q0'][inds]
            params.append(['q',np.round(float(MaxLiSolution['KO_q0']),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
    
    for i in range(nKOs):
        if binary:
            i+=1
        params.append(['','KO_'+str(i+1),'','','',''])
        param = posteriors_single['M_KO_'+str(i)][inds]
        params.append(['M',np.round(float(MaxLiSolution['M_KO_'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['a_KO_'+str(i)][inds]
        params.append(['a',np.round(float(MaxLiSolution['a_KO_'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['KO_P'+str(i)][inds]
        params.append(['P',np.round(float(MaxLiSolution['KO_P'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['KO_K'+str(i)][inds]
        params.append(['K',np.round(float(MaxLiSolution['KO_K'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['KO_ecc'+str(i)][inds]
        params.append(['e',np.round(float(MaxLiSolution['KO_ecc'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['KO_w'+str(i)][inds]
        params.append(['w',np.round(float(MaxLiSolution['KO_w'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_phi'+str(i)][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['KO_phi'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_Tper'+str(i)][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['KO_Tper'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])
    for i in range(numpl):
        inds = indices[i]
        params.append(['','Planet_'+str(i+1),'','','',''])
        param = posteriors_single['M_pl_'][inds]
        params.append(['M',np.round(float(MaxLiSolution['M_pl_'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['a_pl_'][inds]
        params.append(['a',np.round(float(MaxLiSolution['a_pl_'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['P'][inds]
        params.append(['P',np.round(float(MaxLiSolution['P'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['K'][inds]
        params.append(['K',np.round(float(MaxLiSolution['K'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['ecc'][inds]
        params.append(['e',np.round(float(MaxLiSolution['ecc'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['w'][inds]
        params.append(['w',np.round(float(MaxLiSolution['w'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['phi'][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['phi'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['Tper'][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['Tper'+str(i)]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])

    
    parameters = pd.DataFrame(params,columns=param_columns)
    parameters.to_csv(outputloc+system+'_parameters.csv', index=False)

def write_detlim(loc,KM='K'):
    
    if KM =='K':
        K_M = KM+'_ms'
    else:
        K_M = KM+'_M_sun'
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file)
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
