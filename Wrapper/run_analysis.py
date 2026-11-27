#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:06:38 2022

@author: thomasbaycroft
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
# import sys
import time
# import corner
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
from . import make_gaia_datasheets as mgd
from . import GAIAdata_analysis as gda

# import kima


time0 = time.time()

def make_posterior_file_RV(loc,epoch,binary,SB2,star_mass=1.0,star_mass_err=0.0,Yaz_M2=False,M2=0.3,M2_err=0.01,M2_posts=[],phiw=False,Daniel_K2=False,K2=[10,0.1],inc=np.pi/2,inc_err=0):
    #import the posterior samples as a dataframe and add a column which is the likelihood of each posterior sample
    with open(loc+'posterior_sample.txt') as f:
        names = f.readline().split()[1:]
    posterior_file = pd.read_csv(loc+'posterior_sample.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=names)
    
    posterior_info = pd.read_csv(loc+'posterior_sample_info.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=['Level','logL','tiebreak','ID'])
    posterior_file['logL'] = posterior_info['logL']
    
    
    maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file,binary)
   
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
        posterior_file= pfa.add_columns_RV(posterior_file, epoch, binary, Precession, SB2, nKOs, maxNp,star_mass=star_mass,star_mass_err=star_mass_err,Yaz_M2=Yaz_M2,M2=M2,M2_err=M2_err,M2_posts=M2_posts,phiw=phiw,Daniel_K2=Daniel_K2,K2=K2,inc=inc,inc_err=inc_err)
        # a,b=[],[]
        # for i in range(len(posterior_file)):
        #     x,y = op.transit_pred(posterior_file[['M_pri','M_sec','KO_P0','KO_ecc0','Tper_bin','KO_w0','KO_wdot0']].values[i], posterior_file[['M_pl_0','P0','ecc0','Tper0','w0']].values[i], 2458800,epoch)
        #     a.append(x)
        #     b.append(y)
        # posterior_file['transit_pri'],posterior_file['transit_sec'] =a,b
        posterior_file.to_csv(loc+'posterior_file.csv',index=False)

def make_posterior_file_Gaia(loc,epoch=2457936.875,star_mass=1.0,star_mass_err=0.0,P_prior=[10,4000]):
    #import the posterior samples as a dataframe and add a column which is the likelihood of each posterior sample
    with open(loc+'posterior_sample.txt') as f:
        names = f.readline().split()[1:]
    posterior_file = pd.read_csv(loc+'posterior_sample.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=names)
    
    posterior_info = pd.read_csv(loc+'posterior_sample_info.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=['Level','logL','tiebreak','ID'])
    posterior_file['logL'] = posterior_info['logL']
    
    
    maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file,binary=False)
    
    ##############################################################################
    #add columns with other parameters, if this file exists: open it; if not: create it
    #then get maxlisolution and combine the posterior file to have a single planet column
    ##############################################################################
    
    try:
        posterior_file = pd.read_csv(loc+'posterior_file.csv')
    except FileNotFoundError:
        posterior_file= pfa.add_columns_Gaia(posterior_file, epoch, nKOs,maxNp,star_mass=star_mass,star_mass_err=star_mass_err,P_prior=P_prior)
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
     
        posterior_file = posterior_file.reset_index(drop=True)    
    return posterior_file

def read_posteriortxt(loc):
    with open(loc+'posterior_sample.txt') as f:
        names = f.readline().split()[1:]
    posts = pd.read_csv(loc+'posterior_sample.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=names)
    posterior_info = pd.read_csv(loc+'posterior_sample_info.txt',sep='\s+',header=None,skiprows=[0,1],engine='python',names=['Level','logL','tiebreak','ID'])
    posts['logL'] = posterior_info['logL']
    return posts

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
    '''
    Calculates the False Inclusion Periodogram (FIP Hara et al. 2021) from a kima run. Function takes:
        loc: str, path to the directory with the kima samples and posterior samples
        maxNp: int, maximum number of keplerians included in the model
        P_prior: list/array with two values, lower bound and upper bound for the prior used on the orbital period
        dataspan: int, timespan covered by the data in days
    Returns:
        pers: array, period bins used
        fiper_posts: array, FIP values for each period calculated from the posterior samples
        fiper_samps: array, FIP values for each period calculated from the samples
    '''
    
    Pmin, Pmax = P_prior[0], P_prior[1]
    # print(Pmin,Pmax)
    
    #Obtain the Likelihoods and weights for the samples from the kima output files
    Likes_samps = np.array(pd.read_csv(loc+'sample_info.txt',header=None,skiprows=[0],sep='\s+',engine='python',usecols=[1]))
    weights = pd.read_csv(loc+'weights.txt',header=None)
    
    #list of names for columns with the various planets in 
    cols = []
    for i in range(maxNp):
        cols.append('P'+str(i))
        
    #read the period columns from the samples and the posterior samples
    with open(loc+'sample.txt') as f:
        names = f.readline()[1:].split()
    P_samps = np.array(pd.read_csv(loc+'sample.txt',header=None,skiprows=[0],names=names,sep='\s+',engine='python',usecols=cols))
    P_posts = np.array(pd.read_csv(loc+'posterior_sample.txt',skiprows=[0,1],names=names,sep='\s+',engine='python',usecols=cols))
    
    #calculate and normalise the probabilities for each sample/posterior sample
    logL = Likes_samps - np.max(Likes_samps)
    prob_samps = np.exp(logL)*weights
    prob_samps = prob_samps/np.sum(prob_samps)
    prob_posts = np.ones(len(P_posts))/len(P_posts)
    
    #convert the periods to frequencies
    omegas_samps = 2*np.pi/P_samps
    omegas_posts = 2*np.pi/P_posts
    
    freq_int = 2*np.pi/dataspan*1 #frequency resolution

    #define grid of frequencies (The top one is the one I have used mostly, but the bottom is what is recommended in the paper)
    # omegas = np.flip(np.linspace(2*np.pi/Pmax,2*np.pi/Pmin,10000))
    omegas = np.flip(np.linspace(2*np.pi/Pmax,2*np.pi/Pmin,int(5/freq_int)))

    nom = len(omegas)

    tiper_samps = np.zeros(nom)
    tiper_posts = np.zeros(nom)
    
    #calculate the True Inclusion Probabilities (TIP)
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
    
    #FIP is 1-TIP
    fiper_samps = 1-tiper_samps
    fiper_posts = 1-tiper_posts
    pers = 2*np.pi/omegas
    
    return pers, fiper_posts, fiper_samps
        
        
        
def calc_nump_variables(posterior_file,binary=True,loc=None):
    #calculate the Bayes' Factor for each number of planets relative to one less 
    #and therefore the number of planets that we are going to plot
    maxNp = int(posterior_file['maxNp'].iloc[0])
    
    Bayes_factors = np.zeros(maxNp)
    Nps = np.zeros(maxNp+1)
    numpl = 0
    for i in range(maxNp):
        Nps[i] = max(np.sum([posterior_file['Np']==float(i)]),1)
        Bayes_factors[i] = max(np.sum([posterior_file['Np']==float(i+1)]),1)/max(np.sum([posterior_file['Np']==float(i)]),1)
        if Bayes_factors[i] >=7:
            numpl = i+1
    if maxNp>=1:
        i+=1
        Nps[i] = max(np.sum([posterior_file['Np']==float(i)]),1)
    
    Bayes_factors_samps = Bayes_factors
    if loc != None:
        Bayes_factors_samps = Samples_BFs(loc,maxNp)
        
    nKOs = 0
    found = False
    while found == False:
        try:
            P = posterior_file['KO_P'+str(nKOs+int(binary))]
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
        try:
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
        except KeyError: 
            if SB2:
                rvdata_pri = rvdata[['tdb_jd','vrad1','svrad1']]
                rvdata_sec = rvdata[['tdb_jd','vrad2','svrad2']]
                rvdata_sec.columns,rvdata_pri.columns = ['bjd','vrad','svrad'],['bjd','vrad','svrad']
                datas.append(rvdata_pri)
                datas.append(rvdata_sec)
            else:
                if bisFW:
                    rvdata_pri = rvdata[['tdb_jd','vrad','svrad','bis_span','fwhm']]
                    rvdata_pri.columns = ['bjd','vrad','svrad','bis_span','fwhm']
                else:
                    rvdata_pri = rvdata[['tdb_jd','vrad','svrad']]
                    rvdata_pri.columns = ['bjd','vrad','svrad']
                datas.append(rvdata_pri)
        
    return datas
    
 
def make_datasheets(system,outputloc,inputloc,datafiles,instruments,bis_fw=False,SB2=False,binary=False,precession=False,GR=False,Tides=False,studentT=False,drift_order=0,P_prior=[1,2000],CMR=[1.0,1.0],units='kms',enfnp=0):
    
    loc = inputloc
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl,nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file,binary=binary,loc=loc)
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
        # e_hist = np.histogram(posteriors['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
        e_hist = np.histogram(posts['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
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
            MaxLiSolution['M_sec'] = fcs.masses(SB2,float(MaxLiSolution['KO_K0'].iloc[0]),float(MaxLiSolution['KO_P0'].iloc[0]),float(MaxLiSolution['KO_ecc0'].iloc[0]),q=1.0,M1=float(MaxLiSolution['M_pri'].iloc[0]))[1]
    
    #combine the columns for multiple planets into one for periodogram search
    columns_of_planetary_interest = ['P','K','phi','ecc','w','wdot','M_pl_','a_pl_','Tper']
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_','Tper']
    #for posterior file restricted to numpl
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    if numpl>0:
        posteriors_single = posteriors_single[posteriors_single['P']>0.0]
    #for the full posterior file (excluding not found planets)
    full_posteriors = pfa.combine_planets(posterior_file,maxNp,maxNp,columns_of_planetary_interest)
    if maxNp>0:
        full_posteriors = full_posteriors[full_posteriors['P']>0.0]
    
    
    ##############################################################################
    #find peaks in combined periodogram 
    ##############################################################################
    #currently using simple peak finding, could do with refining maybe more variables
    variables = ['P','K']
    variables_2 = ['KO_P','KO_K']
    indices=None
    indices_bin=None
    indices_KO = None
    indices_full=None
    if binary:
        for var in variables_2:
            maxlivar=[]
            maxlivar.append(MaxLiSolution[var+str(0)])
            indices_bin = pf.simple_peaks(posteriors_single, var+str(0), maxlivar, 1, pre_index=indices_bin)
        indices_bin=indices_bin[0]    
        # indices_bin = posteriors_single['KO_P0']!=0
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
        
    if nKOs >=1:
        for var in variables_2:
            maxlivar=[]
            for i in range(nKOs+1):
                maxlivar.append(MaxLiSolution[var+str(i)])
            indices_KO = pf.simple_peaks(posteriors_single, var+str(0), maxlivar, nKOs+1, pre_index=indices_KO)


    
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
        
        
        primary = RV_planets(rvdata_pri,instruments[i],bis_and_FWHM=bis_fw,RV_units=units)
        if len(datafiles)>=2 and i+1<len(datafiles):
            primary.add_offset(MaxLiSolution['offset'+str(i+1)])
        primary.add_planet(numpl,MaxLiSolution)
        primary.add_known_obj(nKOs,MaxLiSolution)
        primary.add_vsys(MaxLiSolution['vsys'])
        if len(datafiles)==1:
            try:
                primary.add_jitter(MaxLiSolution['extra_sigma'])
            except KeyError:
                try:
                    primary.add_jitter(MaxLiSolution['jitter'+str(i+1)])
                except KeyError:
                    primary.add_jitter(MaxLiSolution['jitter'])
        else:
            primary.add_jitter(MaxLiSolution['jitter'+str(i+1)])
        primary.add_drift(drift_order,MaxLiSolution)
        if binary:
            primary.add_binary(binary_params,GR=GR,Tides=Tides,precession=precession,M0=float(MaxLiSolution['M_pri'].iloc[0]),M1=float(MaxLiSolution['M_sec'].iloc[0]),R0=float(MaxLiSolution['R_pri'].iloc[0]))
            if studentT:
                primary.calc_stud_T_outliers(float(MaxLiSolution['nu'].iloc[0]))
            if SB2:
                secondary = RV_planets(rvdata_sec,instruments[i],RV_units=units)
                if len(datafiles)>=2 and i+1<len(datafiles):
                    secondary.add_offset(MaxLiSolution['offset_sec'+str(i+1)])
                secondary.add_binary(binary_params,GR=GR,Tides=Tides,precession=precession,sb2=True,q=float(MaxLiSolution.KO_q0.iloc[0]),M0=float(MaxLiSolution['M_sec'].iloc[0]),M1=float(MaxLiSolution['M_pri'].iloc[0]))
                secondary.add_planet(numpl,MaxLiSolution)
                secondary.add_known_obj(nKOs,MaxLiSolution)
                secondary.add_vsys(MaxLiSolution['vsys_sec'])
                if len(datafiles)==1:
                    try:
                        secondary.add_jitter(MaxLiSolution['extra_sigma_sec'])
                    except KeyError:
                        secondary.add_jitter(MaxLiSolution['jitter_sec'+str(i+1)])
                else:
                    secondary.add_jitter(MaxLiSolution['jitter_sec'+str(i+1)])
                secondary.add_drift(drift_order,MaxLiSolution)
                if studentT:
                    primary.calc_stud_T_outliers(float(MaxLiSolution['nu'].iloc[0]))
                secondaries.append(secondary)
                secondary.write_reses(outputloc+'datafiles/', system,prisec='_sec')
                primary.write_reses(outputloc+'datafiles/', system,prisec='_pri')
            else:
                primary.write_reses(outputloc+'datafiles/', system)
        else:
            if studentT:
                primary.calc_stud_T_outliers(float(MaxLiSolution['nu'].iloc[0]))
            primary.write_reses(outputloc+'datafiles/', system)
        primaries.append(primary)

        
        numdats.append(len(primary.bjds))
    
    mints=[]
    maxts=[]
    times = []
    for pri_data in primaries:
        mints.append(min(pri_data.bjds))
        maxts.append(max(pri_data.bjds))
        times += list(pri_data.bjds)
        
    times = np.sort(times)
    dataspan = float(max(maxts) - min(mints))
    
    #Make the various datasheets
    datasheet_num = 1
    md.make_binary_sheet(system,outputloc,posteriors_single, len_posts, indices, indices_bin, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=precession)
    datasheet_num += 1
    for i in range(nKOs):
        md.make_object_sheet(system,outputloc,True,i,datasheet_num,posteriors_single, indices_KO, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=False)
        datasheet_num += 1
    for i in range(numpl):
        md.make_object_sheet(system,outputloc,False,i,datasheet_num,posteriors_single, indices, full_posteriors, numpl,[Bayes_factors,Nps,Bayes_factors_samps],MaxLiSolution,primaries,secondaries,SB2=SB2,binary=binary,precession=False)
        datasheet_num += 1
    md.make_timeseries_sheet(system,outputloc,datasheet_num,instruments, MaxLiSolution, numpl, nKOs, primaries,sec_data_list=secondaries,binary=binary,SB2=SB2,P_prior=P_prior,drift_order=drift_order)    
    datasheet_num +=1 
    fiptip = FIP_TIP(loc, maxNp, P_prior, dataspan)
    md.make_extras_sheet(system,outputloc,datasheet_num,full_posteriors, indices_full, MaxLiSolution,fiptip,times,primaries,secondaries,binary=binary,P_prior=P_prior,SB2=SB2)
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
    params.append(['vsys',np.round(float(MaxLiSolution['vsys'].iloc[0]),8),np.round(np.percentile(vsyss,50),8),np.round(np.percentile(vsyss,84)-np.percentile(vsyss,50),8),np.round(np.percentile(vsyss,50)-np.percentile(vsyss,16),8),'m/s'])
    if SB2:
        vsys2s = posteriors_single['vsys_sec'][inds]
        params.append(['vsys_sec',np.round(float(MaxLiSolution['vsys_sec'].iloc[0]),1),np.round(np.percentile(vsys2s,50),8),np.round(np.percentile(vsys2s,84)-np.percentile(vsys2s,50),8),np.round(np.percentile(vsys2s,50)-np.percentile(vsys2s,16),8),'m/s'])
    params.append(['reference epoch',round(float(MaxLiSolution['Epoch'].iloc[0]),8),nul,nul,nul,'bjd'])
    params.append(['data timespan',round(dataspan),nul,nul,nul,'days'])
    for i,pri in enumerate(primaries):
        if i >=1:
            param = posteriors_single['offset'+str(i)][inds]
            params.append(['offset',np.round(float(MaxLiSolution['offset'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        if SB2:
            if i >=1:
                param = posteriors_single['offset_sec'+str(i)][inds]
                params.append(['offset_sec',np.round(float(MaxLiSolution['offset_sec'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
            sec = secondaries[i]
            merr2,rms2,jit2 = round(sec.mean_rv_err(),4),round(sec.rms(),4),round(sec.jitter,4)
        else:
            merr2,rms2,jit2 = nul,nul,nul
        params.append([nul,instruments[i],nul,nul,nul,nul])
        params.append(['mean RV_err',round(pri.mean_rv_err(),4),merr2,nul,nul,'m/s'])
        params.append(['RMS',round(pri.rms(),4),rms2,nul,nul,'m/s'])
        params.append(['Jitter',round(pri.jitter,4),jit2,nul,nul,'m/s'])
    drift = ['slope','quadr','cubic']
    for i in range(drift_order):
        param = posteriors_single[drift[i]][inds]
        params.append(['drift_'+str(i+1),np.round(float(MaxLiSolution[drift[i]].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8)])
    for i in range(maxNp):
        params.append(['Bayes Factor '+str(i+1)+'/'+str(i),Bayes_factors[i],Bayes_factors_samps[i],nul,nul])
    if binary:
        params.append(['','Binary','','','',''])
        param = posteriors_single['M_pri'][inds]
        params.append(['M_pri',np.round(float(MaxLiSolution['M_pri'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['M_sec'][inds]
        params.append(['M_sec',np.round(float(MaxLiSolution['M_sec'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        param = posteriors_single['KO_P0'][inds]
        params.append(['P_obs',np.round(float(MaxLiSolution['KO_P0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['a_pri'][inds]
        params.append(['a_pri',np.round(float(MaxLiSolution['a_pri'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['a_sec'][inds]
        params.append(['a_sec',np.round(float(MaxLiSolution['a_sec'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['KO_K0'][inds]
        params.append(['K',np.round(float(MaxLiSolution['KO_K0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['KO_ecc0'][inds]
        params.append(['e',np.round(float(MaxLiSolution['KO_ecc0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['KO_w0'][inds]
        params.append(['w',np.round(float(MaxLiSolution['KO_w0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_phi0'][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['KO_phi0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['Tper_bin'][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['Tper_bin'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])
        if primaries[0].Bin_precession == True:
            param = posteriors_single['KO_wdot0'][inds]
            params.append(['w_dot',np.round(float(MaxLiSolution['KO_wdot0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'arcsec/yr'])
            param = posteriors_single['KO_Pano0'][inds]
            params.append(['P_ano',np.round(float(MaxLiSolution['KO_Pano0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        if SB2:
            param = posteriors_single['KO_q0'][inds]
            params.append(['q',np.round(float(MaxLiSolution['KO_q0'].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
    
    for i in range(nKOs):
        if binary:
            i+=1
        params.append(['','KO_'+str(i+1),'','','',''])
        param = posteriors_single['M_KO_'+str(i)][inds]
        if float(MaxLiSolution['M_KO_'+str(i)].iloc[0]) > 0.08:
            params.append(['M',np.round(float(MaxLiSolution['M_KO_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        else:
            params.append(['M',np.round(float(MaxLiSolution['M_KO_'+str(i)].iloc[0]),8)*Ms/Mj,np.round(np.percentile(param,50),8)*Ms/Mj,np.round(np.percentile(param,84)-np.percentile(param,50),8)*Ms/Mj,np.round(np.percentile(param,50)-np.percentile(param,16),8)*Ms/Mj,'M_jup'])
        param = posteriors_single['a_KO_'+str(i)][inds]
        params.append(['a',np.round(float(MaxLiSolution['a_KO_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['KO_P'+str(i)][inds]
        params.append(['P',np.round(float(MaxLiSolution['KO_P'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['KO_K'+str(i)][inds]
        params.append(['K',np.round(float(MaxLiSolution['KO_K'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['KO_ecc'+str(i)][inds]
        params.append(['e',np.round(float(MaxLiSolution['KO_ecc'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['KO_w'+str(i)][inds]
        params.append(['w',np.round(float(MaxLiSolution['KO_w'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_phi'+str(i)][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['KO_phi'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['KO_Tper'+str(i)][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['KO_Tper'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])
    for i in range(numpl):
        inds = indices[i]
        params.append(['','Planet_'+str(i+1),'','','',''])
        param = posteriors_single['M_pl_'][inds]
        if float(MaxLiSolution['M_pl_'+str(i)].iloc[0]) > 0.08:
            params.append(['M',np.round(float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        elif float(MaxLiSolution['M_pl_'+str(i)].iloc[0]) > 0.0001:
            params.append(['M',np.round(float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),8)*Ms/Mj,np.round(np.percentile(param,50),8)*Ms/Mj,np.round(np.percentile(param,84)-np.percentile(param,50),8)*Ms/Mj,np.round(np.percentile(param,50)-np.percentile(param,16),8)*Ms/Mj,'M_jup'])
        else:
            params.append(['M',np.round(float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),8)*Ms/Me,np.round(np.percentile(param,50),8)*Ms/Me,np.round(np.percentile(param,84)-np.percentile(param,50),8)*Ms/Me,np.round(np.percentile(param,50)-np.percentile(param,16),8)*Ms/Me,'M_earth'])

        param = posteriors_single['a_pl_'][inds]
        params.append(['a',np.round(float(MaxLiSolution['a_pl_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'AU'])
        param = posteriors_single['P'][inds]
        params.append(['P',np.round(float(MaxLiSolution['P'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'days'])
        param = posteriors_single['K'][inds]
        params.append(['K',np.round(float(MaxLiSolution['K'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'m/s'])
        param = posteriors_single['ecc'][inds]
        params.append(['e',np.round(float(MaxLiSolution['ecc'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),''])
        param = posteriors_single['w'][inds]
        params.append(['w',np.round(float(MaxLiSolution['w'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['phi'][inds]
        params.append(['phi_0',np.round(float(MaxLiSolution['phi'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'rad'])
        param = posteriors_single['Tper'][inds]
        params.append(['T_per',np.round(float(MaxLiSolution['Tper'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'bjd'])

    
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

def make_rvgaia_datasheets(system,outputloc,inputloc,datafile,Jitter=True,studentT=False,enfnp=0.1,P_prior=[10,5000],model=None,res=None,acceleration=False,jerk=False,scan_dep_signal=False,n_scan_dep_components=0,star_mass=1.0,Gaia_epoch=2457936.875):
    make_gaia_datasheets(system,outputloc,inputloc,datafile,Jitter,studentT,enfnp,P_prior,model,res,acceleration,jerk,scan_dep_signal,n_scan_dep_components,star_mass,Gaia_epoch)


def make_gaia_datasheets(system,outputloc,inputloc,datafile,Jitter=True,studentT=False,enfnp=0.1,P_prior=[10,5000],model=None,res=None,acceleration=False,jerk=False,scan_dep_signal=False,n_scan_dep_components=0,star_mass=1.0,Gaia_epoch=2457936.875):
    loc = inputloc
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, nKOs, Bayes_factors, Nps, Bayes_factors_samps = calc_nump_variables(posterior_file,binary=False,loc=loc)
    BFs_nps = [Bayes_factors,Nps,Bayes_factors_samps]
    if enfnp >=1:
        numpl = enfnp
    
    #now only consider the posteriors with the correct number of planets
    len_posts = len(posterior_file)
    posteriors = posterior_file[posterior_file['Np']==numpl].reset_index(drop=True)
    
    #Get the maximum likelihood solution
    posts = posteriors.copy()
    #first without any cuts, the true maximum likelihood solution
    MaxLiSolution_nocut = posts[posts['logL']==max(posts['logL'])].head(1)
    posteriors_np = posteriors.copy()
    posteriors_np = posteriors_np[posteriors_np.Np == numpl]
    
    mask = np.ones(len(res.sample))
    #then if ecc not distinct from zero in the posteriors, choose the maximum likelihood with ecc<0.1
    for i in range(numpl):
        # e_hist = np.histogram(posteriors['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
        e_hist = np.histogram(posts['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
        if np.argmax(e_hist) in [0,1,2]:
            emax = 0.1
        else:
            emax = 1.0
        posts = posts[posts['ecc'+str(i)]<=emax]
        if res is not None:
            mask = np.logical_and(mask,res.sample.T[res.indices['planets.e']][i] < emax)
    
    MaxLiSolution = posts[posts['logL']==max(posts['logL'])].head(1)

    # MaxLiSolution['Mstar'] = np.mean(posteriors.Mstar) #enforce ML star mass to be the mean value of distribution
    
    if res is not None:
        if numpl>=1:
            pykima_ML = res.maximum_likelihood_sample(mask=mask,Np=numpl,printit=False)
        else:
            pykima_ML = res.maximum_likelihood_sample(Np=0,printit=False)
    else:
        pykima_ML = None

    #combine the columns for multiple planets into one for periodogram search
    columns_of_planetary_interest = ['P','phi','ecc','a','w', 'cosi','W','M_pl_','sma_pl_','inc','Tper']
    #for posterior file restricted to numpl
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    if numpl>0:
        posteriors_single = posteriors_single[posteriors_single['P']>0.0]
    #for the full posterior file (excluding not found planets)
    full_posteriors = pfa.combine_planets(posterior_file,maxNp,maxNp,columns_of_planetary_interest)
    if maxNp>0:
        full_posteriors = full_posteriors[full_posteriors['P']>0.0]

    ##############################################################################
    #find peaks in combined periodogram 
    ##############################################################################
    #currently using simple peak finding, could do with refining maybe more variables
    variables = ['P','a']
    indices=None
    indices_bin=None
    indices_KO = None
    indices_full=None
    
    for var in variables:
        maxlivar=[]
        for i in range(numpl):
            maxlivar.append(MaxLiSolution[var+str(i)])
        indices = pf.simple_peaks(posteriors_single, var, maxlivar, numpl, pre_index=indices)
        indices_full = pf.simple_peaks(full_posteriors, var, maxlivar, numpl, pre_index=indices_full)
            
    ##############################################################################
    #Get data and calculate dataspan
    ##############################################################################

    data = pd.read_csv(datafile,header=0,skiprows=[1],names=['times','w','w_err','psi','pf'],sep='\t',usecols=[0,1,2,3,4])
    dataspan = data.times[len(data.times)-1] - data.times[0]

    data_kepmodel_format = pd.read_csv(datafile,header=0,skiprows=[1],names=['obs_time_tcb','centroid_pos_al','centroid_pos_error_al','scan_pos_angle','parallax_factor_al'],sep='\t',usecols=[0,1,2,3,4])


    ##############################################################################
    #Caluclate FIP
    ##############################################################################

    fiptip = FIP_TIP(loc, maxNp, P_prior, dataspan)    

    ##############################################################################
    #Get fit/noise params
    ##############################################################################
    round_val = 6

    if res is not None:
        res_alscan, res_alpha, res_delta = gda.astrometry_residuals(res,pykima_ML)
    else:
        res_alscan = 0.0
    Ndata = len(data)
    RMS = np.round(np.sqrt(np.sum(np.array(res_alscan)**2)/len(res_alscan)),round_val)
    Median_err = np.round(np.median(data.w_err),round_val)
    Mean_err = np.round(np.mean(data.w_err),round_val)
    scan_angles = data.psi.copy()
    angle_hist = np.histogram(scan_angles,bins=np.linspace(-np.pi,np.pi,21))[0]
    fracscanbins = np.sum(angle_hist>0)/20
    propinmax = np.round(np.max(angle_hist)/Ndata,round_val)

    Nparams = 5
    if Jitter:
        Nparams += 1
    if studentT:
        Nparams += 1
    if acceleration:
        Nparams += 2
    if jerk:
        Nparams += 2
    Nparams += 7*nKOs + 7*numpl
    if scan_dep_signal:
        Nparams += n_scan_dep_components*2

    deg_free = Ndata - Nparams
    Chi_squared = np.round(np.sum((np.array(res_alscan)/np.array(data.w_err))**2),round_val)
    Chi_squared_red = np.round(np.sum((np.array(res_alscan)/np.array(data.w_err))**2)/deg_free,round_val)

    ML_jitter = np.round(float(MaxLiSolution.jitter.iloc[0]),round_val)

    noise_params = [Ndata,np.round(dataspan,round_val),ML_jitter,RMS,Median_err,Mean_err,Chi_squared,Chi_squared_red,fracscanbins,propinmax]
    noise_params_names = ['# of data = ','dataspan = ', 'MaxLi Jitter = ','Residual RMS = ','Median error = ','Mean error = ',r'$\chi^{2}$ = ',r'$\chi^{2}_{\nu}$ = ',r'Fraction of $\psi$ bins '+'\ncovered = ', r'Proportion of $\psi$ in '+'\nmaximum bin (of 20) = ']
    
    if studentT:
        noise_params.append(float(MaxLiSolution.nu.iloc[0]))
        noise_params_names.append(r'student-t $\nu$ = ')

    ##############################################################################
    #Orbit SNRs
    ##############################################################################
    SNRs = []
    SNR_names = []
    for i in range(nKOs):
        SNR = fcs.astrometric_SNR(float(MaxLiSolution.Mstar.iloc[0]),float(MaxLiSolution['M_KO_'+str(i)].iloc[0]),float(MaxLiSolution['KO_P'+str(i)].iloc[0]),float(1000/MaxLiSolution.parallax.iloc[0]),Beta=0,Nobs=Ndata,sigma=Median_err*1000)
        SNRs.append(SNR)
        SNR_names.append('SNR_KO_'+str(i))
    
    for i in range(numpl):
        SNR = fcs.astrometric_SNR(float(MaxLiSolution.Mstar.iloc[0]),float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),float(MaxLiSolution['P'+str(i)].iloc[0]),float(1000/MaxLiSolution.parallax.iloc[0]),Beta=0,Nobs=Ndata,sigma=Median_err*1000)
        SNRs.append(SNR)
        SNR_names.append('SNR_pl_'+str(i))

    ##############################################################################
    #Set names and titles for parameters
    ##############################################################################
    columns_baseline = ['da','dd','mua','mud','parallax']
    names_baseline = [r'$\Delta\alpha_{\star}$',r'$\Delta\delta$',r'$\mu_{\alpha_{\star}}$',r'$\mu_{\delta}$',r'$\varpi$']
    units_baseline = [' [mas]',' [mas]',' [mas/yr]',' [mas/yr]',' [mas]']
    value_mult = [1.0,1.0,1.0,1.0,1.0]
    if acceleration:
        columns_baseline = ['da','dd','mua','mud','accela','acceld','parallax']
        names_baseline = [r'$\Delta\alpha_{\star}$',r'$\Delta\delta$',r'$\mu_{\alpha_{\star}}$',r'$\mu_{\delta}$',r'$\ddot{\alpha_{\star}}$',r'$\ddot{\delta}$',r'$\varpi$']
        units_baseline = [' [mas]',' [mas]',' [mas/yr]',' [mas/yr]',r' [mas/yr$^2$]',r' [mas/yr$^2$]',' [mas]']
        value_mult = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    if jerk:
        columns_baseline = ['da','dd','mua','mud','accela','acceld','jerka','jerkd','parallax']
        names_baseline = [r'$\Delta\alpha_{\star}$',r'$\Delta\delta$',r'$\mu_{\alpha_{\star}}$',r'$\mu_{\delta}$',r'$\ddot{\alpha_{\star}}$',r'$\ddot{\delta}$',r'$\dddot{\alpha_{\star}}$',r'$\dddot{\delta}$',r'$\varpi$']
        units_baseline = [' [mas]',' [mas]',' [mas/yr]',' [mas/yr]',r' [mas/yr$^2$]',r' [mas/yr$^2$]',r' [mas/yr$^3$]',r' [mas/yr$^3$]',' [mas]']
        value_mult = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    

    columns_of_KO_interest = ['KO_P','KO_phi','KO_ecc','KO_a','KO_w', 'KO_cosi','KO_W','M_KO_','sma_KO_','KO_inc','KO_Tper']
    KO_param_names = [r'KO_$P$',r'KO_$\phi$',r'KO_$e$',r'KO_$a$',r'KO_$\omega$',r'KO_$\cos{i}$',r'KO_$\Omega$',r'$M$_KO_',r'$a$_KO_',r'KO_$i$',r'KO_$T_{\rm peri}$']
    planet_param_names = [r'$P$',r'$\phi$',r'$e$',r'$a$',r'$\omega$',r'$\cos{i}$',r'$\Omega$',r'$M$_pl_',r'$a$_pl_',r'$i$',r'$T_{\rm peri}$']
    param_units = [' [days]',' [rad]',' ',' [mas]',' [rad]',' ',' [rad]',r' [$M_{\odot}$]',' [AU]',' [deg]',' ']

    ML_param_cols = columns_baseline.copy()
    ML_param_names = names_baseline.copy()
    ML_param_units = units_baseline.copy()
    KO_mass_islow = np.zeros(nKOs)
    for i in range(nKOs):
        ML_param_cols += [columns_of_KO_interest[j]+str(i) for j in range(len(columns_of_KO_interest))]
        ML_param_names += [KO_param_names[j]+str(i) for j in range(len(KO_param_names))]
        if float(MaxLiSolution['M_KO_'+str(i)].iloc[0]) < 0.06:
            value_mult += [1.0,1.0,1.0,1.0,1.0,1.0,1.0,fcs.Msun/fcs.Mj,1.0,1.0,1.0]
            units_new = [' [days]',' [rad]',' ',' [mas]',' [rad]',' ',' [rad]',r' [$M_{\rm Jup}$]',' [AU]',' [deg]',' ']
            ML_param_units += units_new
            KO_mass_islow[i] = 1
        else:
            value_mult += [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            ML_param_units += param_units
    pl_mass_islow = np.zeros(numpl)
    for i in range(numpl):
        ML_param_cols += [columns_of_planetary_interest[j]+str(i) for j in range(len(columns_of_planetary_interest))]
        ML_param_names += [planet_param_names[j]+str(i) for j in range(len(planet_param_names))]
        if float(MaxLiSolution['M_pl_'+str(i)].iloc[0]) < 0.06:
            value_mult += [1.0,1.0,1.0,1.0,1.0,1.0,1.0,fcs.Msun/fcs.Mj,1.0,1.0,1.0]
            units_new = [' [days]',' [rad]',' ',' [mas]',' [rad]',' ',' [rad]',r' [$M_{\rm Jup}$]',' [AU]',' [deg]',' ']
            ML_param_units += units_new
            pl_mass_islow[i] = 1
        else:
            value_mult += [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            ML_param_units += param_units
    
    ##############################################################################
    #Make datasheets
    ##############################################################################
    mgd.make_baseline_sheet(system, outputloc, posteriors, len_posts, indices, indices_full, posteriors_np, posteriors_single, full_posteriors, nKOs, numpl, BFs_nps, fiptip, res, pykima_ML, MaxLiSolution, ML_param_names, ML_param_cols, ML_param_units, value_mult, dataspan, noise_params, noise_params_names, acceleration, jerk)

    fap = mgd.make_extras_sheet(system,outputloc,posteriors,full_posteriors,indices_full,numpl,data_kepmodel_format,nKOs,star_mass,float(MaxLiSolution.parallax.iloc[0]),dataspan,Gaia_epoch)

    logZ = mgd.make_diagnostics_sheet(inputloc,outputloc,system)

    
    ##############################################################################
    #Make residuals file
    ##############################################################################
    residual_file = data.copy()
    residual_file['residuals'] = res_alscan

    if studentT:
        Gd = norm(loc=0, scale=np.hypot(residual_file.w_err, ML_jitter)).pdf(res_alscan)
        Td = t(df=MaxLiSolution.nu, loc=0, scale=np.hypot(residual_file.w_err, ML_jitter)).pdf(res_alscan)
        studT_ratios = Td/Gd
        for j in range(len(Td)):
            if Td[j] == 0. and Gd[j] == 0.:
                studT_ratios[j] = 1.
        residual_file['Tratio'] = studT_ratios

    residual_file.to_csv(outputloc+system+'_4_residuals.csv', index=False)


    ##############################################################################
    #Make a table of parameters
    ##############################################################################
    nul='---'
    param_columns = ['parameter','MaxLi_value','Median_value','+','-','units']
    params = []
    params.append(['logZ',logZ,nul,nul,nul,nul])
    #Add noise/data parameters
    noise_params_units = [nul,'days','mas','mas','mas','mas',nul,nul,nul,nul]
    if studentT:
        noise_params_units.append(nul)
    param = posteriors['jitter']
    add_parameter_row(params,MaxLiSolution,param,'jitter','mas')
    dummy = 0
    for name,param,unit in zip(noise_params_names,noise_params,noise_params_units):
        if dummy == 0 or dummy == 1:
            dummy += 1
            params.append([name[:-2],param,nul,nul,nul,unit])  
        elif dummy == 2:
            dummy = 3
        else:
            params.append([name[:-2],param,nul,nul,nul,unit])    
    #Add planet stats
    params.append(['number of known-objects',nKOs,nul,nul,nul,nul])
    params.append(['number of planets',numpl,nul,nul,nul,nul])
    for i in range(len(SNRs)):
        params.append([SNR_names[i],SNRs[i],nul,nul,nul,nul])
    params.append(['Kepmodel peak FAP',fap,nul,nul,nul,nul])
    for i in range(maxNp):
        params.append(['Bayes Factor '+str(i+1)+'/'+str(i),Bayes_factors[i],Bayes_factors_samps[i],nul,nul])

    params.append(['','Star parameters','','','',''])
    for variable,units in zip(['Mstar'],['M_sun']):
        param = posteriors[variable]
        add_parameter_row(params,MaxLiSolution,param,variable,units) 

    params.append(['','baseline astrometry','','','',''])
    for variable,units in zip(columns_baseline,units_baseline):
        param = posteriors[variable]
        add_parameter_row(params,MaxLiSolution,param,variable,units) 

    if scan_dep_signal:
        params.append(['','scan-dependent signal','','','',''])
        for i in range(n_scan_dep_components):
            variable = 'Amp'+str(2*i +3)
            units = 'mas'
            param = posteriors[variable]
            add_parameter_row(params,MaxLiSolution,param,variable,units)
            variable = 'theta'+str(2*i +3)
            units = 'nul'
            param = posteriors[variable]
            add_parameter_row(params,MaxLiSolution,param,variable,units)

    for i in range(nKOs):
        params.append(['','Known-object_'+str(i),'','','',''])
        if KO_mass_islow[i]:
            param = posteriors['M_KO_'+str(i)]*fcs.Msun/fcs.Mj
            params.append(['KO_M'+str(i),np.round(float(MaxLiSolution['M_KO_'+str(i)].iloc[0]),8)*fcs.Msun/fcs.Mj,np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_Jup'])
        else:
            param = posteriors['M_KO_'+str(i)]
            params.append(['KO_M'+str(i),np.round(float(MaxLiSolution['M_KO_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        variable_list = ['KO_P'+str(i),'KO_phi'+str(i),'KO_ecc'+str(i),'KO_a'+str(i),'KO_w'+str(i), 'KO_cosi'+str(i),'KO_W'+str(i),'sma_KO_'+str(i),'KO_inc'+str(i),'KO_Tper'+str(i)]
        units_list = ['days','rad',nul,'mas','rad',nul,'rad','AU','deg',nul]
        for variable,units in zip(variable_list,units_list):
            param = posteriors[variable]
            add_parameter_row(params,MaxLiSolution,param,variable,units) 

    for i in range(numpl):
        inds = indices[i]
        params.append(['','Planet_'+str(i),'','','',''])
        if pl_mass_islow[i]:
            param = posteriors_single['M_pl_'][inds]*fcs.Msun/fcs.Mj
            params.append(['M'+str(i),np.round(float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),8)*fcs.Msun/fcs.Mj,np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_Jup'])
        else:
            params.append(['M'+str(i),np.round(float(MaxLiSolution['M_pl_'+str(i)].iloc[0]),8),np.round(np.percentile(param,50),8),np.round(np.percentile(param,84)-np.percentile(param,50),8),np.round(np.percentile(param,50)-np.percentile(param,16),8),'M_sun'])
        variable_list = ['P','phi','ecc','a','w', 'cosi','W','sma_pl_','inc','Tper']
        units_list = ['days','rad',nul,'mas','rad',nul,'rad','AU','deg',nul]
        for variable,units in zip(variable_list,units_list):
            param = posteriors_single[variable][inds]
            add_parameter_row(params,MaxLiSolution,param,variable+str(i),units) 

    parameters = pd.DataFrame(params,columns=param_columns)
    parameters.to_csv(outputloc+system+'_5_parameters.csv', index=False)

def add_parameter_row(params,MaxLiSolution,param,variable,units,dps=8):
    params.append([variable,np.round(float(MaxLiSolution[variable].iloc[0]),dps),np.round(np.percentile(param,50),dps),np.round(np.percentile(param,84)-np.percentile(param,50),dps),np.round(np.percentile(param,50)-np.percentile(param,16),dps),units])
