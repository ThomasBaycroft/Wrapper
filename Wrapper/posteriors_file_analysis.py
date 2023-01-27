#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:42:58 2023

@author: thomasbaycroft
"""
import pandas as pd
from . import functions as fcs
import numpy as np


def add_columns(posterior_file,epoch,binary,Precession,SB2,nKOs,maxNp,star_mass=1.0):
    '''
    takes a dataframe for the posteriors and adds columns for various derived quantities etc
    requires various flags/values:
        epoch: refernce epoch for calculations (BJD)
        binary: is this a binary (boolean)
        Precession: is apsidal precession of binary included (boolean)
        SB2: is the binary double lined (boolean)
        nKOs: number of known objects (int)
        maxNp: maximum number of planets searched for (int)
        star_mass: mass of central star (M_sun) used if not SB2
    saves the resulting file as a csv and also returns it as a dataframe
    '''
    #add columns with other parameters, masses, au, P_anom, wdot_theory, T0 (...)
    posterior_file['Epoch'] = epoch
    if binary:
        if SB2:
            q = posterior_file['KO_q0']
        else:
            q = 1.0
        #calculate the mass of the primary and secondary star if a binary
        posterior_file['M_pri'],posterior_file['M_sec'] = fcs.masses(SB2,posterior_file['KO_K0'],posterior_file['KO_P0'],posterior_file['KO_ecc0'],q=q,M1=star_mass)
        posterior_file['a_pri'],posterior_file['a_sec'] = fcs.sma_2body(posterior_file['KO_P0'],posterior_file['M_pri'],posterior_file['M_sec'])
        #calculate the GR precession values and initalise the dynamical wdot column 
        posterior_file['wdot_GR'] = fcs.wdot_GR(posterior_file['M_pri'],posterior_file['M_sec'],posterior_file['KO_ecc0'], posterior_file['KO_P0'])
        posterior_file['wdot_dyn'] = 0.
        #calculate the time of periastron for the binary
        posterior_file['Tper_bin'] = posterior_file['Epoch'] - posterior_file['KO_P0']*posterior_file['KO_phi0']/2/np.pi
        if nKOs >1:
            for i in range(1,nKOs):
                #calculate the mass of any additional known objects
                posterior_file['M_KO_'+str(i)] = fcs.masses(False,posterior_file['KO_K'+str(i)],posterior_file['KO_P'+str(i)],posterior_file['KO_ecc'+str(i)])[1]
                posterior_file['a_KO_'+str(i)] = fcs.sma_2body(posterior_file['KO_P'+str(i)],posterior_file['M_pri']+posterior_file['M_sec'],posterior_file['M_KO_'+str(i)])[1]
                #add the dynamical wdot contribution
                wdotvals = fcs.wdot_dyn(posterior_file['M_pri'],posterior_file['M_sec'], posterior_file['M_KO_'+str(i)], posterior_file['KO_ecc0'],  posterior_file['KO_ecc'+str(i)],  posterior_file['KO_P0'],  posterior_file['KO_P'+str(i)],  posterior_file['KO_w0'], 0.)
                posterior_file['wdot_dyn'] += wdotvals.fillna(0)
                #calculate the time of periastron for each known object
                posterior_file['KO_Tper'+str(i)] = posterior_file['Epoch'] - posterior_file['KO_P'+str(i)]*posterior_file['KO_phi'+str(i)]/2/np.pi
    
        for i in range(maxNp):
            #get indices of where no planet is found for this column to make the masses zero
            isplanet = posterior_file['P'+str(i)]>0.0
            #calculate masses of planets in each sample, assume a central mass of the sum of the binary masses
            posterior_file['M_pl_'+str(i)] = fcs.masses(False,posterior_file['K'+str(i)],posterior_file['P'+str(i)],posterior_file['ecc'+str(i)],M1=posterior_file['M_pri']+posterior_file['M_sec'])[1]*isplanet
            posterior_file['a_pl_'+str(i)] = fcs.sma_2body(posterior_file['P'+str(i)],posterior_file['M_pri']+posterior_file['M_sec'],posterior_file['M_pl_'+str(i)])[1]*isplanet
            #add the dynamical wdot contribution
            wdotvals = fcs.wdot_dyn(posterior_file['M_pri'],posterior_file['M_sec'], posterior_file['M_pl_'+str(i)], posterior_file['KO_ecc0'],  posterior_file['ecc'+str(i)],  posterior_file['KO_P0'],  posterior_file['P'+str(i)],  posterior_file['KO_w0'], 0.)
            posterior_file['wdot_dyn'] += wdotvals.fillna(0)*isplanet
            #calculate the time of periastron for each planetary solution
            posterior_file['Tper'+str(i)] = posterior_file['Epoch'] - posterior_file['P'+str(i)]*posterior_file['phi'+str(i)]/2/np.pi
    
        if Precession:
            posterior_file['KO_Pano0'] = fcs.Period_change(posterior_file['KO_P0'], posterior_file['KO_ecc0'], posterior_file['KO_wdot0'])
    else:
        posterior_file['M_star'] = star_mass
        for i in range(nKOs):
            #calculate the mass of any additional known objects
            posterior_file['M_KO_'+str(i)] = fcs.masses(False,posterior_file['KO_K'+str(i)],posterior_file['KO_P'+str(i)],posterior_file['KO_ecc'+str(i)])[1]
            posterior_file['a_KO_'+str(i)] = fcs.sma_2body(posterior_file['KO_P'+str(i)],posterior_file['M_pri']+posterior_file['M_sec'],posterior_file['M_KO_'+str(i)])[1]
            #calculate the time of periastron for each known object
            posterior_file['KO_Tper'+str(i)] = posterior_file['Epoch'] - posterior_file['KO_P'+str(i)]*posterior_file['KO_phi'+str(i)]/2/np.pi

        for i in range(maxNp):
            #get indices of where no planet is found for this column to make the masses zero
            isplanet = posterior_file['P'+str(i)]>0.0
            #calculate masses of planets in each sample, assume a central mass of the sum of the binary masses
            posterior_file['M_pl_'+str(i)] = fcs.masses(False,posterior_file['K'+str(i)],posterior_file['P'+str(i)],posterior_file['ecc'+str(i)],M1=posterior_file['M_star'])[1]*isplanet
            posterior_file['a_pl_'+str(i)] = fcs.sma_2body(posterior_file['P'+str(i)],posterior_file['M_star'],posterior_file['M_pl_'+str(i)])[1]*isplanet
            #calculate the time of periastron for each planetary solution
            posterior_file['Tper'+str(i)] = posterior_file['Epoch'] - posterior_file['P'+str(i)]*posterior_file['phi'+str(i)]/2/np.pi



    return posterior_file

def combine_planets(posterior_file,maxNp,numpl,columns_of_planetary_interest):
    
    # singlePosteriors = pd.DataFrame(names=)
    columns = list(posterior_file.columns)
    for i in range(maxNp):
        for j in range(len(columns_of_planetary_interest)):
            columns.remove(columns_of_planetary_interest[j] + str(i))
    dataframes=[]        
    final_cols = columns+columns_of_planetary_interest
    for i in range(numpl):
        df= posterior_file[columns + [columns_of_planetary_interest[j] + str(i) for j in range(len(columns_of_planetary_interest))]]
        df.columns=final_cols
        dataframes.append(df)
    final = pd.concat([dataframes[i] for i in range(numpl)], ignore_index=True,names=final_cols)    
    
    return final

