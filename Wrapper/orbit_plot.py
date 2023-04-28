#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:13:36 2022

@author: thomasbaycroft
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as cst
import time

Ms = cst.M_sun.value
AU = cst.au.value
G = cst.G.value
Mj = cst.M_jup.value

def semi_major_axis(M_tot,P):
    return ((P*24*3600)**2*G*(M_tot)*Ms/(4*np.pi**2))**(1/3)/AU

def plot_orbits(Masses,Periods,eccs,omegas,ax=None,fig=None,circumbinary=False,SB2=False,timeseries=[],tpers=[0,0,0],alpha=1,text=True,cs=['r','b','g','m','c','darkviolet','orange'],ls='-'):
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
        if len(timeseries)>0:
            nus = phase(np.array(timeseries),Periods[i],tpers[i])*2*np.pi
            theta_points = nus +omegas[i] - np.pi/2
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
        
   
def plot_circumbinary(Bin_P,M1,M2,Bin_e,Bin_w,pl_Ps,pl_Ms,pl_es,pl_ws,timeseries=[],tpers=[],SB2=False,ax=None,fig=None,alpha=1,text=True,c=['r','b','g','m','c','darkviolet','orange'],ls='-'):
    pl_Ms = [mass*Mj/Ms for mass in pl_Ms]
    masses = np.array([M1,M2]+pl_Ms)
    Periods = np.array([Bin_P,Bin_P]+pl_Ps)
    eccs = np.array([Bin_e,Bin_e]+pl_es)
    omegas = np.array([Bin_w-np.pi,Bin_w]+pl_ws)
    plot_orbits(masses,Periods,eccs,omegas,circumbinary=True,timeseries=timeseries,tpers=tpers,SB2=SB2,ax=ax,fig=fig,alpha=alpha,text=text,cs=c,ls=ls)
    
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
# bin_params = [1.2388,1.20295,19.657878,0.448178,10,1.231,0.]
# pl_params = [2.31*Mj/Ms,143.09,0.14,2,4]
# times = np.linspace(0,500,500)
# time1=time.time()
# a,b = transit_pred(bin_params,pl_params,2,0)
# print(a,b)
# print(time.time()-time1)