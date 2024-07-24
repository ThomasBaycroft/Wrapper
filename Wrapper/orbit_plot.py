#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:13:36 2022

@author: thomasbaycroft
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as cst
from astropy.time import Time

import time
import pandas as pd
from . import functions as fcs
from . import run_analysis as ra 
from . import posteriors_file_analysis as pfa
from . import peak_finder as pf

Ms = cst.M_sun.value
AU = cst.au.value
G = cst.G.value
Mj = cst.M_jup.value

def semi_major_axis(M_tot,P):
    return ((P*24*3600)**2*G*(M_tot)*Ms/(4*np.pi**2))**(1/3)/AU

def plot_orbits(Masses,Periods,eccs,omegas,ax=None,fig=None,circumbinary=False,SB2=False,timeseries=[],tpers=[0,0,0],alpha=1,text=True,cs=['r','b','g','m','c','darkviolet','orange'],ls='-',bords=False):
    '''
    Plot orbits on a polar plot, showing where each RV datapoint has been sampled coloured by time, takes:
        Masses: list, Mass of each body (Msun)
        Periods: list, orbital periods (days)
        eccs:  list, eccentricities
        omegas: list, arguments of pericentre
        ax: an axis on which to plot (if not provided creates one)
        fig: a figure on which to plot (if not provided creates one)
        circumbinary: bool, circumbinary flag
        SB2: bool, SB2 flag
        timeseries: list, RV times
        tpers: list, times of periastron (defaults to all being zero and assuming length three)
    '''
    M_tot = np.sum(Masses)
    smas = semi_major_axis(M_tot,Periods)
    numbod = len(Masses)
    if ax==None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    ax.set_rticks([])
    ax.grid(False)
    
    theta = np.arange(0,2*np.pi,0.01)
    
    rs = []
    
    if circumbinary==True and SB2==False:
        start = 1
    else:
        start=0
    
    for i in range(0,len(smas)):
        r = smas[i]*(1-eccs[i]**2)/(1+eccs[i]*np.cos(theta-omegas[i]+np.pi/2))
        rs.append(r)
    for i in range(start,len(smas)):   
        r = rs[i]
        ax.plot(theta, r,alpha=alpha,c=cs[i],ls=ls)
        if bords:
            rM = np.ones(len(r))*max(r)
            rm = np.ones(len(r))*min(r)
            ax.plot(theta, rM,c=cs[i],ls=':',alpha=0.4)
            ax.plot(theta, rm,c=cs[i],ls=':',alpha=0.4)
        if len(timeseries)>0:
            Ms = phase(np.array(timeseries),Periods[i],tpers[i])*2*np.pi
            Es = fcs.ecc_anomaly(Ms, eccs[i])
            nus = fcs.true_anomaly(Es, eccs[i])
            theta_points = nus + omegas[i] - np.pi/2
            r_points = smas[i]*(1-eccs[i]**2)/(1+eccs[i]*np.cos(nus))
            c = ax.scatter(theta_points,r_points,c=timeseries,cmap='cool',alpha=alpha)
            if i == 0:
                fig.colorbar(c, label='bjd')
            

    ax.set_rmin(0)
    if circumbinary==True and numbod>=3 and text==True:
        min1 = min(rs[2]/rs[0])
        ax.text(3*np.pi/2,max(r)*1.3,'min dist ratio to primary ='+str(round(min1,2))+'  ',ha='right')
        min2 = min(rs[2]/rs[1])
        ax.text(3*np.pi/2,max(r)*1.3,'min dist ratio to secondary ='+str(round(min2,2))+'  ',ha='left')
        
   
def plot_circumbinary(Bin_P,M1,M2,Bin_e,Bin_w,pl_Ps,pl_Ms,pl_es,pl_ws,timeseries=[],tpers=[],SB2=False,ax=None,fig=None,alpha=1,text=True,c=['r','b','g','m','c','darkviolet','orange'],ls='-',bords=False):
    pl_Ms = [mass*Mj/Ms for mass in pl_Ms]
    masses = np.array([M1,M2]+pl_Ms)
    Periods = np.array([Bin_P,Bin_P]+pl_Ps)
    eccs = np.array([Bin_e,Bin_e]+pl_es)
    omegas = np.array([Bin_w-np.pi,Bin_w]+pl_ws)
    plot_orbits(masses,Periods,eccs,omegas,circumbinary=True,timeseries=timeseries,tpers=tpers,SB2=SB2,ax=ax,fig=fig,alpha=alpha,text=text,cs=c,ls=ls,bords=bords)
    
def phase(t, P, T0, phi_max=1):
    """
    Returns an array of phase values for the input timeseries
    
    parameters:
        t       : array, list
                  timeseries of your data
        P       : float, int
                  period to phase data onto
        T0      : float, int
                  reference time for phase=0
        phi_max : float, int
                  maximum phase value - phase will be in the range [-phi_max, phi_max]

    returns:
        phi     : array
                  phase values of t
    """
    phi = ((t - T0) % P) / P 
    idx = np.where(phi >= phi_max)
    try:
        phi[idx] -= 1
    except:
        pass
    return phi

# plot_circumbinary(19.657878,1.2388,1.20295,0.448178,1.231,[143.09],[2.31],[0.14],[4],timeseries=[0,1,3,5,8,12,13,15,18,20,23,25],tpers=[0,0,0])
def x_coord(phi,a,e,w):
    
    return -a*(1-e**2)*np.cos(w+phi)/(1+e*np.cos(phi))

def y_coord(phi,a,e,w):
    
    return a*(1-e**2)*np.sin(w+phi)/(1+e*np.cos(phi))
    
def get_anomalies(time,P,Tper,e):
    
    M = 2.*np.pi*(time-Tper)/P
    E = ecc_anomaly(M,e)
    phi = true_anomaly(E,e)
    
    return phi    
    
def true_anomaly(E, e):
    return 2. * np.arctan( np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.))

def ecc_anomaly(M, e):
    M = np.atleast_1d(M)
    E0 = M; E = M
    for _ in range(200):
        g = E0 - e * np.sin(E0) - M
        gp = 1.0 - e * np.cos(E0)
        E = E0 - g / gp

        # check for convergence
        if (np.linalg.norm(E - E0, ord=1) <= 1.234e-10): 
            return E
        # keep going
        E0 = E

    # no convergence, return the best estimate
    return E

def transitability_pred(bin_params,pl_params,start_time,epoch,prec=0.1,ax=''):
    '''
    bin_params: M0, M1, P, e, T_per, w, wdot
    pl_params: M, P, e, T_per, w

    '''
    
    Mbin = bin_params[0] + bin_params[1]
    Pbin = bin_params[2]
    
    times = np.arange(start_time,start_time + pl_params[1],prec)
    
    sma_pri = semi_major_axis(Mbin,Pbin)*bin_params[1]/Mbin
    sma_sec = semi_major_axis(Mbin,Pbin)*bin_params[0]/Mbin
    
    Ms_bin = phase(np.array(times),bin_params[2],bin_params[4])*2*np.pi
    Es_bin = fcs.ecc_anomaly(Ms_bin, bin_params[3])
    nus_bin = fcs.true_anomaly(Es_bin, bin_params[3])
    thetas = nus_bin + bin_params[5] - np.pi/2
    r_pri = sma_pri*(1-bin_params[3]**2)/(1+bin_params[3]*np.cos(nus_bin))
    r_sec = sma_sec*(1-bin_params[3]**2)/(1+bin_params[3]*np.cos(nus_bin))
    
    xs_pri = r_pri*np.sin(thetas-np.pi)
    xs_sec = r_sec*np.sin(thetas)
    xmax, xmin = max(max(xs_pri),max(xs_sec)), min(min(xs_pri),min(xs_sec))


    sma_pl =  semi_major_axis(Mbin+pl_params[0],pl_params[1])*Mbin/(Mbin+pl_params[0])

    Ms_pl = phase(np.array(times),pl_params[1],pl_params[3])*2*np.pi
    Es_pl = fcs.ecc_anomaly(Ms_pl, pl_params[2])
    nus_pl = fcs.true_anomaly(Es_pl, pl_params[2])
    thetas_pl = nus_pl + pl_params[4] - np.pi/2
    r_pl = sma_pl*(1-pl_params[2]**2)/(1+pl_params[2]*np.cos(nus_pl))
    
    xs = r_pl*np.sin(thetas_pl)
    ys = r_pl*np.cos(thetas_pl)
    
    ind = np.logical_and(np.logical_and(xs<=xmax,xs>=xmin),ys>=0)
    xs_in = xs[np.logical_and(xs<=xmax,xs>=xmin)]
    rs_in = r_pl[ind]
    thetas_in = thetas_pl[ind]
    times_in = times[ind]
    
    if ax =='':
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.set_rticks([])
    ax.grid(False)
    
    ax.plot(thetas-np.pi,r_pri)
    ax.plot(thetas,r_sec)
    
    ax.plot(thetas_pl,r_pl,c='grey',alpha=0.1)
    ax.plot(thetas_in,rs_in,c='g',alpha=0.2)
    
    ax.set_rmin(0)
    
    return [times_in[0],times_in[-1]]

def transitability_pred_posts(loc,start_time,num=100,cluster=True):
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    # maxNp, numpl,nKOs, Bayes_factors, Nps, Bayes_factors_samps = ra.calc_nump_variables(posterior_file,loc)
    numpl=1
    maxNp = 3
    binary=True
    posts = posterior_file[posterior_file.Np == numpl]
    
    if cluster:
        MaxLiSolution_nocut = posts[posts['logL']==max(posts['logL'])].head(1)
        
        #then if ecc not distinct from zero in the posteriors, choose the maximum likelihood with ecc<0.1
        for i in range(numpl):
            e_hist = np.histogram(posts['ecc'+str(i)],bins = np.linspace(0,1,100))[0]
            if np.argmax(e_hist) in [0,1,2]:
                emax = 0.1
            else:
                emax = 1.0
            posts = posts[posts['ecc'+str(i)]<=emax]
        
        MaxLiSolution = posts[posts['logL']==max(posts['logL'])].head(1)
        
        columns_of_planetary_interest = ['P','K','phi','ecc','w','wdot','M_pl_','a_pl_','Tper']
        columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_','Tper']
        #for posterior file restricted to numpl
        posteriors_single = pfa.combine_planets(posts, maxNp, numpl, columns_of_planetary_interest)
        
        
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

    posts = posteriors_single.loc[indices[0]]
    
    posts = posts.reset_index(drop=True)
    
    rand = np.random.randint(0,len(posts),num)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    t0,t1=[],[]
    
    for i in rand:
        bin_params = posts.loc[i][['M_pri','M_sec','KO_P0','KO_ecc0','Tper_bin','KO_w0','KO_wdot0']]
        if cluster:
            pl_params = posts.loc[i][['M_pl_','P','ecc','Tper','w']]
        else:
            pl_params = posts.loc[i][['M_pl_0','P0','ecc0','Tper0','w0']]
        epoch = posts.loc[i].Epoch
        tims = transitability_pred(bin_params, pl_params, start_time, epoch,ax=ax)
        t0.append(tims[0])
        t1.append(tims[1])
    
    print('Transitable window:')
    print('start (bjd):',np.median(t0),'+',np.percentile(t0,84)-np.median(t0),'-',np.median(t0)-np.percentile(t0,16))
    print('end (bjd):',np.median(t1),'+',np.percentile(t1,84)-np.median(t1),'-',np.median(t1)-np.percentile(t1,16))

def transit_pred(bin_params,pl_params,start_time,epoch,dataframe=False):
    '''
    bin_params: M0, M1, P, e, T_per, w, wdot
    pl_params: M, P, e, T_per, w
    '''
    # if dataframe:
    #     Mbin = bin_params['M_pri'] + bin_params['M_sec']
    #     Pbin = bin_params['KO_P0']
    #     ebin = bin
    Mbin = bin_params[0] + bin_params[1]
    Pbin = bin_params[2]
    ebin = bin_params[3]
    Tperbin = bin_params[4]
    wsec0 = bin_params[5]
    try:
        wdotbin = bin_params[6]*np.pi/3600/365.25/180
    except IndexError:
        wdotbin = 0.
    P_bin = Period_change(Pbin, ebin, wdotbin)
    
    a_0 = semi_major_axis(Mbin,Pbin)*bin_params[1]/Mbin
    a_1 = semi_major_axis(Mbin,Pbin)*bin_params[0]/Mbin
    
    Ppl = pl_params[1]
    epl = pl_params[2]
    Tperpl = pl_params[3]
    wpl = pl_params[4]
    a_pl = semi_major_axis(Mbin+pl_params[0],Ppl)
    
    times = np.array([start_time,start_time + Pbin])
    wsec = wsec0 + wdotbin*(times-epoch)
    phi_pl = get_anomalies(times,Ppl,Tperpl,epl)
    phi_bin = get_anomalies(times,Pbin,Tperbin,ebin)
    x_pri = x_coord(phi_bin,a_0,ebin,wsec-np.pi)
    x_pl = x_coord(phi_pl,a_pl,epl,wpl)
    Dx = x_pri-x_pl
    y_pl = y_coord(phi_pl[0],a_pl,epl,wpl)
    while Dx[0]*Dx[1]>=0 or y_pl <= 0:
        times = times + Pbin
        wsec = wsec0 + wdotbin*(times-epoch)
        phi_pl = get_anomalies(times,Ppl,Tperpl,epl)
        phi_bin = get_anomalies(times,Pbin,Tperbin,ebin)
        x_pri = x_coord(phi_bin,a_0,ebin,wsec-np.pi)
        x_pl = x_coord(phi_pl,a_pl,epl,wpl)
        Dx = x_pri-x_pl
        y_pl = y_coord(phi_pl[0],a_pl,epl,wpl)
   
    dt = times[1]-times[0]
    newtime=np.mean(times)
    while abs(dt)>=0.01:
        newtime = np.mean(times)
        phi_p = get_anomalies(newtime,Ppl,Tperpl,epl)
        phi_b = get_anomalies(newtime,Pbin,Tperbin,ebin)
        wsecnew = wsec0 + wdotbin*(newtime-epoch)
        delx = x_coord(phi_b,a_0,ebin,wsecnew-np.pi) - x_coord(phi_p,a_pl,epl,wpl)
        if delx*Dx[0]>=0:
            times[0] = newtime
        else :
            times[1] = newtime
        wsec = wsec0 + wdotbin*(times-epoch)
        phi_pl = get_anomalies(times,Ppl,Tperpl,epl)
        phi_bin = get_anomalies(times,Pbin,Tperbin,ebin)
        x_pri = x_coord(phi_bin,a_0,ebin,wsec-np.pi)
        x_pl = x_coord(phi_pl,a_pl,epl,wpl)
        Dx = x_pri-x_pl
        dt = times[1]-times[0]
        
    Pri_transit_time = newtime
    times[0] = Pri_transit_time - Pbin
    times[1] = Pri_transit_time + Pbin
    wsec = wsec0 + wdotbin*(times-epoch)
    phi_pl = get_anomalies(times,Ppl,Tperpl,epl)
    phi_bin = get_anomalies(times,Pbin,Tperbin,ebin)
    x_sec = x_coord(phi_bin,a_1,ebin,wsec)
    x_pl = x_coord(phi_pl,a_pl,epl,wpl)
    Dx = x_sec-x_pl
    
    dt = times[1]-times[0]
    newtime=np.mean(times)
    while abs(dt)>=0.01:
        newtime = np.mean(times)
        wsecnew = wsec0 + wdotbin*(newtime-epoch)
        phi_p = get_anomalies(newtime,Ppl,Tperpl,epl)
        phi_b = get_anomalies(newtime,Pbin,Tperbin,ebin)
        delx = x_coord(phi_b,a_1,ebin,wsecnew) - x_coord(phi_p,a_pl,epl,wpl)
        if delx*Dx[0]>=0:
            times[0] = newtime
        else :
            times[1] = newtime
        wsec = wsec0 + wdotbin*(times-epoch)
        phi_pl = get_anomalies(times,Ppl,Tperpl,epl)
        phi_bin = get_anomalies(times,Pbin,Tperbin,ebin)
        x_sec = x_coord(phi_bin,a_1,ebin,wsec)
        x_pl = x_coord(phi_pl,a_pl,epl,wpl)
        Dx = x_sec-x_pl
        dt = times[1]-times[0]
        
    Sec_transit_time = newtime
    
    return Pri_transit_time, Sec_transit_time 

def Period_change(P2,e,wdot):
    '''
    Gives the actual period since wdot overestimates by a slight factor
    We take off from the period, the time taken due to the extra apsidal motion

    Assume spread evenly/circular orbit
    '''
    return P2*(1+wdot/(3600*180*365.25*2)*P2)    

def eclipse_range(P,T0,t_range=[],P_err=0,T0_err=0,num=1000, night=True):
    
    if len(t_range) == 0:
        t_range = [T0,T0+100]
    if len(t_range) == 1:
        t_range.append(t_range[0]+100)
    
    Ps = np.random.randn(num)*P_err + P
    T0s = np.random.randn(num)*T0_err + T0
    
    epochs = []
    epoch = 0
    t = T0
    while t <= t_range[1]:
        t += P
        epoch += 1
        if t>=t_range[0] and t <= t_range[1]:
            epochs.append(epoch)

    Times = np.ndarray([len(epochs),num])
    for i,epo in enumerate(epochs):
        Times[i,:] = T0s + epo*Ps
    
    print('There are',str(len(epochs)),'eclipses in range, those during nights are:')
    for i in range(len(epochs)):
        time = Time(np.median(Times[i,:]),format='jd')
        plus = (np.percentile(Times[i,:],84)-np.median(Times[i,:]))*24*60
        minus = (np.median(Times[i,:])-np.percentile(Times[i,:],16))*24*60
        if time.ymdhms.hour >20 or time.ymdhms.hour<7 or night==False:
            print('bjd = ',np.median(Times[i,:]),'+',np.percentile(Times[i,:],84)-np.median(Times[i,:]),'-',np.median(Times[i,:])-np.percentile(Times[i,:],16))    
            print('UT time of median is',time.iso,'+',plus,'-',minus,'mins')

def obs_time_density(ax,orb,nbins,today,data_times):
    
    nuhist,Tbins = orb.Tbins_in_trueanom(today,nbins,data_times)
    
    ax.stairs(nuhist[0], Tbins, fill=True)


# bin_params = [1.2388,1.20295,19.657878,0.448178,10,1.231,0.]
# pl_params = [2.31*Mj/Ms,143.09,0.14,2,4]
# times = np.linspace(0,500,500)
# time1=time.time()
# a,b = transit_pred(bin_params,pl_params,2,0)
# print(a,b)
# print(time.time()-time1)