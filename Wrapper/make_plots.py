#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:08:32 2023

@author: thomasbaycroft
"""

import matplotlib.pyplot as plt
import numpy as np
from . import functions as fcs
from . import run_analysis as ra
from . import make_plots as mplt
from . import posteriors_file_analysis as pfa
import astropy.constants as cst
import pandas  as pd
import random

Ms = cst.M_sun.value
Mj = cst.M_jup.value
Msat = 5.683*10**26
Mn = 1.024*10**26
Me = cst.M_earth.value

def plot_rv_curves(pri_data_list,sec_data_list=[],colours=['b','r','g','m','purple','orange'],inflate_errors=False):
    '''
    Plots a vertical stack of subplots, the phased RV curves for each body, as well as the residuals
    in a timeseries, given:
        pri_data_list: list of 'RV_planets' objects for data on the primary star
        sec_data_list same as pri_data_list but for secondary star if this is an SB2
        colours: list of colours to plot sequential instruments
    outputs the plot, use function below to add an individual plot to a larger ensemble
    '''
    
    sec_fill = False
    
    num = pri_data_list[0].num_bodies -1 
    fig, ax = plt.subplots(num,1,tight_layout=True,figsize=(6,num*8))
    tot_phas = []
    tot_rv = []
    tot_err = []
    for i,data in enumerate(pri_data_list):
        lines = i==0
        j = 0
        # if data.binary:
        #     plot_rv_orbit(ax[j],data.binary_orbit,data,'binary',lines=lines,colour=colours[i],inflate_errors=inflate_errors)
        #     j+=1
        for k in range(data.numpl):
            plot_rv_orbit(ax,data.planet_orbits[k],data,0,lines=lines,colour=colours[i],inflate_errors=inflate_errors)
            # plot_rv_orbit(ax[j],data.planet_orbits[k],data,0,lines=lines,colour=colours[i],inflate_errors=inflate_errors)
            j+=1
            tot_rv += list(data.rv_curve_planet(0))
            tot_phas += list(data.planet_orbits[k].phase(data.bjds))
            tot_err += list(data.svrad)
        ax.set(xlabel='phase')
        # ax[j-1].set(xlabel='phase')
        # plot_rv_orbit(ax[j],data.planet_orbits[k],data,'residuals_long',lines=lines,colour=colours[i],inflate_errors=inflate_errors)
    
    if i==0:
        colours[0] = colours[1]
        sec_fill=True
    
    for i,data in enumerate(sec_data_list):
        lines = i==0
        j = 0
        # if data.binary:
        #     plot_rv_orbit(ax[j],data.binary_orbit,data,'binary',lines=lines,fill=sec_fill,colour=colours[i],inflate_errors=inflate_errors)
        #     j+=1
        for k in range(data.numpl):
            plot_rv_orbit(ax,data.planet_orbits[k],data,0,lines=False,fill=sec_fill,colour=colours[i],inflate_errors=inflate_errors)
            # plot_rv_orbit(ax[j],data.planet_orbits[k],data,0,lines=False,fill=sec_fill,colour=colours[i],inflate_errors=inflate_errors)
            j+=1
            tot_rv += list(data.rv_curve_planet(0))
            tot_phas += list(data.planet_orbits[k].phase(data.bjds))
            tot_err += list(data.svrad)
        # plot_rv_orbit(ax[j],data.planet_orbits[k],data,'residuals_long',lines=lines,fill=sec_fill,colour=colours[i],inflate_errors=inflate_errors)
    
    # plot_rv_phase_binned(ax,np.array(tot_phas),np.array(tot_rv),np.array(tot_err))    
    # plot_rv_phase_binned(ax[1],np.array(tot_phas),np.array(tot_rv),np.array(tot_err))
   
    return fig, ax

def plot_rv_orbit(ax,orb,data,body,colour='b',fill=True,lines=True,er_bar_col='grey',inflate_errors=False,bin_phase=False):
    '''
    plots a single RV curve on an axis given:
        ax: the ax object on which to plot
        orb: the 'orbit' object to plot
        data: the 'RV_planet' object to get the rvs from
        body: whether a binary, a planet or residuals being plotted
        colour: colour of scatter point
        fill: whether to fill the points
        lines: whether to plot the rv_model lines
        er_bar_col: colour for the errorbars
        inflate_error: bool; whether to make the errorbars the quadratic sum of uncertainty and jitter
        bin_phase: whether to also plot the data binned into 10 phases to help guide the eye
    '''
    
    if fill:
        fc = colour
    else:
        fc='none'
        
    if inflate_errors:
        erbars = (data.svrad**2 + data.jitter**2)**(1/2)
    else:
        erbars = data.svrad
    
    times = np.linspace(min(data.bjds),max(data.bjds),20000)
    if body == 'binary':
        # plot_rv_phase_binned(ax,orb.phase(data.bjds),data.rv_curve_binary()/1000)
        ax.errorbar(orb.phase(data.bjds),data.rv_curve_binary()/1000,yerr=erbars/1000,fmt='none',c=er_bar_col,alpha=0.7)
        ax.errorbar(orb.phase(data.bjds),data.rv_curve_binary()/1000,fmt='.',c=colour,mfc=fc,markersize=7.)
        ax.errorbar(orb.phase(data.bjds)-1,data.rv_curve_binary()/1000,yerr=erbars/1000,fmt='.',c='grey',alpha=0.6)
        ax.errorbar(orb.phase(data.bjds)+1,data.rv_curve_binary()/1000,yerr=erbars/1000,fmt='.',c='grey',alpha=0.6)
        if lines:
            ax.plot(np.sort(orb.phase(times)),orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))]/1000,c='black')
            ax.plot(np.sort(orb.phase(times))-1,orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))]/1000,c='grey',alpha=0.6)
            ax.plot(np.sort(orb.phase(times))+1,orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))]/1000,c='grey',alpha=0.6)
        ax.axhline(0,c='grey',ls='--')
        ax.set(xlim=[-0.1,1.1],ylabel=r'RV (km s$^{-1}$)')
    elif body == 'binary_unphased':
        sub = 2400000
        ax.errorbar(data.bjds-sub,data.rv_curve_binary()/1000,yerr=erbars/1000,fmt='none',c=er_bar_col,alpha=0.7)
        ax.errorbar(data.bjds-sub,data.rv_curve_binary()/1000,fmt='.',c=colour,mfc=fc)
        if lines:
            ax.axhline(0,c='black',alpha=0.6)
            ax.plot(times-sub,orb.RV_keplerian_add(times)/1000,c=colour,alpha=0.7)
        ax.set(ylabel=r'RV km s$^{-1}$')
    elif body == 'residuals_unphased':
        sub = 2400000
        ax.errorbar(data.bjds-sub,data.residuals()/1000,yerr=erbars/1000,fmt='none',c=er_bar_col,alpha=0.7)
        ax.errorbar(data.bjds-sub,data.residuals()/1000,fmt='.',c=colour,mfc=fc)
        if lines:
            ax.axhline(0,c='black',alpha=0.6)
        m = max(-min(data.residuals()),max(data.residuals()))*1.1/1000
        ax.set(ylim=[-m,m],ylabel = 'Residuals')#,xlabel='Time (bjd-'+str(sub)+')')
        
    elif body == 'residuals':
        ax.errorbar(orb.phase(data.bjds),data.residuals()/1000,yerr=erbars/1000,fmt='none',c=er_bar_col,alpha=0.7)
        ax.errorbar(orb.phase(data.bjds),data.residuals()/1000,fmt='.',c=colour,mfc=fc)
        ax.errorbar(orb.phase(data.bjds)-1,data.residuals()/1000,yerr=erbars/1000,fmt='.',c='grey',alpha=0.6)
        ax.errorbar(orb.phase(data.bjds)+1,data.residuals()/1000,yerr=erbars/1000,fmt='.',c='grey',alpha=0.6)
        if lines:
            ax.axhline(0,c='black',alpha=0.6)
        m = max(-min(data.residuals()),max(data.residuals()))*1.1/1000
        ax.set(xlim=[-0.1,1.1],ylim=[-m,m],ylabel=r'res (km s$^{-1}$)')#,xlabel='phase')
        
    else:
        # plot_rv_phase_binned(ax,orb.phase(data.bjds),data.rv_curve_planet(body))
        ax.errorbar(orb.phase(data.bjds),data.rv_curve_orbit(orb),yerr=erbars,fmt='none',c=er_bar_col,alpha=0.7)
        ax.errorbar(orb.phase(data.bjds),data.rv_curve_orbit(orb),fmt='.',c=colour,mfc=fc,alpha=1.,markersize=7.)
        ax.errorbar(orb.phase(data.bjds)-1,data.rv_curve_orbit(orb),yerr=erbars,fmt='.',c='grey',alpha=0.6)
        ax.errorbar(orb.phase(data.bjds)+1,data.rv_curve_orbit(orb),yerr=erbars,fmt='.',c='grey',alpha=0.6)
        if lines:
            ax.plot(np.sort(orb.phase(times)),orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))],c='black',alpha=0.6)
            ax.plot(np.sort(orb.phase(times))-1,orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))],c='grey',alpha=0.6)
            ax.plot(np.sort(orb.phase(times))+1,orb.RV_keplerian_add(times)[np.argsort(orb.phase(times))],c='grey')
        ax.axhline(0,c='grey',ls='--')
        ax.set(xlim=[-0.1,1.1],ylabel=r'RV (m s$^{-1}$)')
        
def plot_rv_phase_binned(ax,phase,rvdata,rverrs,numbins=10,mark='8',colour='black'):
    
    phase = np.array(phase)
    bins = np.linspace(0,1,numbins+1)
    xax = []
    xerrs=[]
    points = []
    errs = []
    for i in range(numbins):
        ind = np.logical_and(phase>=bins[i],phase<bins[i+1])
        xax.append(np.mean(phase[ind]))
        xerrs.append(np.std(phase[ind])/np.sqrt(sum(ind)))
        points.append(np.mean(rvdata[ind]))
        errs.append(np.std(rvdata[ind])/np.sqrt(sum(ind)))
        # errs.append(np.mean(rverrs[ind]))
    ax.errorbar(xax,points,xerr=xerrs,yerr=errs,fmt=mark,c=colour,mfc='none',markersize=10.,markeredgewidth=1.3)
                                 

def plot_detection_limit(ax,posterior_file,KM='K',M_central=1.,colourmap='Greys',detlim_col='purple',numbins=50,lowlim=-1,uplim=None,label=''):
    '''
    plots detection limit give:
        ax: object on which to plot
        posterior_file: posterior_file with a single column for planets with the number removed (ie treated with the pfa.combine_planets function)
        KM: y_axis is K or M
        M_central: central mass of system (M_sun)
        colourmap: colourmap for the hexbin
        detlim_col: coloour for the detection limit line
        numbins: number of period bins for the hexplot
        lowlim;uplim: lower and upper limit on the y-axis variable (logged)
    '''
    ylab = r'Semi-amplitude (m s$^{-1}$)'
    mult=1.
    if KM != 'K':
        KM = 'M_pl_'
        ylab = 'Mass (M_earth)'
        mult = Ms/Me
        if uplim==None:
            uplim = np.log10(max(posterior_file[KM]*mult))
            lowlim = 0.
    
    if label=='':
        label = '99% confidence detection limit'
    
    bincentres, KM99s, p_bins, histvals = make_detection_limit(posterior_file,numbins,KM=KM,lowlim=lowlim)
    
    if uplim == None:
        uplim = np.log10(max(posterior_file[KM]))
    ax.plot(bincentres,KM99s*mult,c=detlim_col,label=label)
    xax = np.array([min(posterior_file['P']),max(posterior_file['P'])])
    if colourmap!=None:
        ax.hexbin(posterior_file['P'],posterior_file[KM]*mult,xscale='log',bins='log',yscale='log', cmap=colourmap,alpha=0.65, gridsize=numbins+1, lw=0, mincnt=1,extent=(np.log10(min(posterior_file['P'])),np.log10(max(posterior_file['P'])),lowlim,uplim))
    
        if KM == 'K':
            Masses, names = [80*Mj/Ms,13*Mj/Ms,Mj/Ms,Msat/Ms,Mn/Ms,2*Me/Ms,Me/Ms],['Stellar_limit','Brown-dwarf limit','Jupiter','Saturn','Neptune','2xEarth','Earth']
            for i, mass in enumerate(Masses):
                plot_planet_line(ax,M_central,mass,xax,text=names[i])
            
    ax.set(xlim=xax,ylim=[10**lowlim,10**uplim],xlabel='Period (days)',ylabel=ylab)
    
    
def make_detection_limit(posterior_file,numbins,KM='K',lowlim=-1,p_prior=[]):
    '''
    given:
        posterior_file: posterior_file with a single column for planets with the number removed (ie treated with the pfa.combine_planets function)
        numbins: number of period bins to take (equally log-spaced)
        KM: whether to perform the detection limit in K or M
        lowlim: lower_limit we are using for KM (log10)
    '''
    if KM != 'K':
        KM = 'M_pl_'
    if p_prior == []:
        bins = np.logspace(np.log10(min(posterior_file['P'])),np.log10(max(posterior_file['P'])),numbins+1)
    else:
        bins = np.logspace(np.log10(p_prior[0]),np.log10(p_prior[1]),numbins+1)
    p_bins = np.histogram(posterior_file['P'],bins=bins)[1]
    histvals = np.histogram(posterior_file['P'],bins=bins)[0]
    KM99s=[]
    bincentres=[]
    for i in range(numbins):
        KMs = posterior_file[np.logical_and(posterior_file['P']>=p_bins[i],posterior_file['P']<p_bins[i+1])][KM]
        if len(KMs) == 0:
            KMs=[10**lowlim]
        KM99s.append(np.percentile(KMs,99))
        bincentres.append(10**((np.log10(p_bins[i])+np.log10(p_bins[i+1]))/2))
        
    # df = pd.DataFrame(np.array([bincentres,KM99s]).T,columns=['P','Detlim_'+KM])
    # df.to_csv('Detection_limit_'+KM+'.csv',index=False)
    return bincentres, np.array(KM99s), p_bins, histvals

def plot_planet_line(ax,M0,Mp,xax,text='',colour='black',M0p=None,M0m=None):
    '''
    plots the line on P-K detlim where a body of given mass would lie:
        ax: object on which we plot
        M0: central mass (M_sun)
        Mp: body mass (M_sun)
        xax: x-axis start and endpoint
        text: what to write next to the line
        colour: colour of the line
        M0p,M0m: plus and minus on the central mass, to give shaded region (M_sun)
    '''
    
    ax.plot(xax, fcs.semi_amplitude(M0, Mp, xax, 0),ls='--',c=colour)
    if M0p != None:
        ax.fill_between(xax,fcs.semi_amplitude(M0+M0m, Mp, xax, 0),fcs.semi_amplitude(M0+M0p, Mp, xax, 0), facecolor=colour,alpha = 0.3)
    ax.annotate(text,(xax[-1], fcs.semi_amplitude(M0, Mp, xax[-1], 0)),textcoords="offset points",xytext=(-10,10),ha='right')

def plot_detlim_lines(ax,loc,KM='K'):
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, Bayes_factors, Nps = ra.calc_nump_variables(posterior_file)
    #now only consider the posteriors with the correct number of planets
    posteriors = posterior_file[posterior_file['Np']==numpl]
    
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_']
    posteriors_single = pfa.combine_planets(posteriors, maxNp, numpl, columns_of_planetary_interest)
    
    xax, detlim, pbins, hvals = mplt.make_detection_limit(posteriors_single,20,KM=KM)
    
    ax.plot(xax,detlim,label=loc)
    
def plot_wdot_planets_loc(ax,loc,detlimloc, number,detlimcol='purple',scattercol='cornflowerblue',outcol='grey'):
    
    
    detlim_posts  = pd.read_csv(detlimloc+'posterior_file.csv')
    maxNp, numpl, nKOs, Bayes_factors, Nps, BFsamps = ra.calc_nump_variables(detlim_posts)
    #now only consider the posteriors with the correct number of planets
    detlim_posteriors = detlim_posts[detlim_posts['Np']==numpl]
    
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_']
    detlim_ps = pfa.combine_planets(detlim_posteriors, maxNp, numpl, columns_of_planetary_interest)
    numbins=50
    xax, detlim,p_bins, hvals = mplt.make_detection_limit(detlim_ps,numbins,KM='M')
    
    posterior_file = pd.read_csv(loc+'posterior_file.csv')
    maxNp, numpl, nKOs, Bayes_factors, Nps, BFsamps = ra.calc_nump_variables(posterior_file)
    #now only consider the posteriors with the correct number of planets
    posteriors = posterior_file[posterior_file['Np']==numpl]
    
    # number = len(posteriors)
    
    P = np.random.rand(number)*980 + 20
    wdot = np.array(posteriors['KO_wdot0'])
    M_under  = []
    P_under = []
    inds = random.sample(list(posteriors.index),number)
    fails=[]
    for j,i in enumerate(inds):
        M = fcs.planet_mass(posteriors['M_pri'][i],posteriors['M_sec'][i],posteriors['KO_ecc0'][i],0,posteriors['KO_P0'][i],P[j],posteriors['KO_w0'][i],0,posteriors['KO_wdot0'][i])
        # M = fcs.planet_mass(np.array(posteriors['M_pri'])[i],np.array(posteriors['M_sec'])[i],np.array(posteriors['KO_ecc0'])[i],0,np.array(posteriors['KO_P0'])[i],P[j],np.array(posteriors['KO_w0'])[i],0,wdot[i])
        # plt.scatter(P[i],M*Ms/Mj,c='cornflowerblue',s=8,alpha=0.04)
        try:
            P_stab=fcs.get_HW_stab(posteriors['M_pri'][i],posteriors['M_sec'][i],posteriors['KO_P0'][i],posteriors['KO_ecc0'][i],ext=True)
            Mdl = detlim[list(np.histogram(P[j],bins=p_bins)[0]).index(1)]
            if M<Mdl and P[j] > P_stab:
                # M_under.append(M*Ms/Mj)
                plt.scatter(P[j],M*Ms/Mj,c=scattercol,s=8,alpha=0.4)
                # P_under.append(P[i])
            else:
                if outcol=='grey':
                    al=0.1
                else:
                    al=0.4
                plt.scatter(P[j],M*Ms/Mj,c=outcol,s=8,alpha=al)
        except ValueError:
            fails.append(P[j])
    print(len(fails),'errors recorded')
        
    plt.plot(xax,detlim*Ms/Mj,c=detlimcol,label='Detection limit')    
    # bincen, perc = get_percentile(np.array(P_under),np.array(M_under),bins=np.linspace(min(P_under),max(P_under),20),percentile=99,log=False)
    # plt.plot(bincen,perc)
    # sns.kdeplot(P_under,M_under,levels=[0.64,0.95],log_scale=(False,False),color='indianred')
    plt.xlabel('Period (days)')
    plt.ylabel(r'Mass (M$_J$)')
    plt.ylim([0.1,1000])
    plt.semilogy()
    # xax = np.linspace(40,840,60)
    # Mbin = np.ones(len(xax))*(2.151+1.992)
    # es = np.ones(len(xax))*0.2419
    # Ks = np.ones(len(xax))*5000
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='b',label='5 km/s')
    # Ks = np.ones(len(xax))*1000
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='r',label='1 km/s')
    # Ks = np.ones(len(xax))*400
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='g',label='400 m/s')
    plt.axhline(80,ls='--',c='grey')
    plt.axhline(13,ls='--',c='grey')
    plt.axvline(42.8,c='g',label='stability limit')
    plt.legend()
    
def plot_wdot_planets_file(ax,posts,detlimloc, number,detlimcol='purple',scattercol='cornflowerblue',outcol='grey',stabcol='r',numbins=50):
    
    
    detlim_posts  = pd.read_csv(detlimloc+'posterior_file.csv')
    maxNp, numpl, nKOs, Bayes_factors, Nps, BFsamps = ra.calc_nump_variables(detlim_posts)
    #now only consider the posteriors with the correct number of planets
    detlim_posteriors = detlim_posts[detlim_posts['Np']==numpl]
    
    columns_of_planetary_interest = ['P','K','phi','ecc','w','M_pl_','a_pl_']
    detlim_ps = pfa.combine_planets(detlim_posteriors, maxNp, numpl, columns_of_planetary_interest)
    numbins=numbins
    xax, detlim,p_bins, hvals = mplt.make_detection_limit(detlim_ps,numbins,KM='M')
    
    posteriors = posts
    
    # number = len(posteriors)
    
    P = np.random.rand(number)*400 + 20
    inds = random.sample(list(posteriors.index),number)
    # print(inds)
    # print(posteriors)
    fails=[]
    P_stabs=[]
    done = 0
    for j,i in enumerate(inds):
        M = fcs.planet_mass(posteriors['M_pri'][i],posteriors['M_sec'][i],posteriors['KO_ecc0'][i],0,posteriors['KO_P0'][i],P[j],posteriors['KO_w0'][i],0,posteriors['KO_wdot0'][i])
        # M = fcs.planet_mass(np.array(posteriors['M_pri'])[i],np.array(posteriors['M_sec'])[i],np.array(posteriors['KO_ecc0'])[i],0,np.array(posteriors['KO_P0'])[i],P[j],np.array(posteriors['KO_w0'])[i],0,wdot[i])
        # plt.scatter(P[i],M*Ms/Mj,c='cornflowerblue',s=8,alpha=0.04)
        try:
            P_stab=fcs.get_HW_stab(posteriors['M_pri'][i],posteriors['M_sec'][i],posteriors['KO_P0'][i],posteriors['KO_ecc0'][i],ext=True)
            P_stabs.append(P_stab)
            try:
                Mdl = detlim[list(np.histogram(P[j],bins=p_bins)[0]).index(1)]
            except ValueError:
                Mdl = 0
            if M<Mdl and P[j] > P_stab:
                # M_under.append(M*Ms/Mj)
                if done >0:
                    plt.scatter(P[j],M*Ms/Mj,c=scattercol,s=8,alpha=0.2)
                else:
                    plt.scatter(P[j],M*Ms/Mj,c=scattercol,s=8,alpha=0.2,label=r'Planet ($\dot{\omega}$)')
                    done = 1
                # P_under.append(P[i])
            else:
                if outcol=='grey':
                    al=0.1
                else:
                    al=0.2
                plt.scatter(P[j],M*Ms/Mj,c=outcol,s=8,alpha=al)
        except ValueError:
            fails.append(P[j])
    print(len(fails),'errors recorded')
        
    plt.plot(xax,detlim*Ms/Mj,c=detlimcol,label='Detection limit')    
    # bincen, perc = get_percentile(np.array(P_under),np.array(M_under),bins=np.linspace(min(P_under),max(P_under),20),percentile=99,log=False)
    # plt.plot(bincen,perc)
    # sns.kdeplot(P_under,M_under,levels=[0.64,0.95],log_scale=(False,False),color='indianred')
    plt.xlabel('Period (days)')
    plt.ylabel(r'Mass (M$_J$)')
    plt.ylim([0.1,1000])
    plt.semilogy()
    # xax = np.linspace(40,840,60)
    # Mbin = np.ones(len(xax))*(2.151+1.992)
    # es = np.ones(len(xax))*0.2419
    # Ks = np.ones(len(xax))*5000
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='b',label='5 km/s')
    # Ks = np.ones(len(xax))*1000
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='r',label='1 km/s')
    # Ks = np.ones(len(xax))*400
    # ms = np.array(fcs.get_mass(Ks,Mbin,xax,es))*Ms/Mj
    # plt.plot(xax,ms,c='g',label='400 m/s')
    # plt.axhline(80,ls='--',c='grey')
    # plt.axhline(13,ls='--',c='grey')
    plt.axvline(np.median(P_stabs),c=stabcol,label='stability limit',ls='--')
    plt.fill_betweenx([0.1,1000],[np.median(P_stabs)],fc='orange',alpha=0.4,label='unstable region')
    plt.legend()    
    
    