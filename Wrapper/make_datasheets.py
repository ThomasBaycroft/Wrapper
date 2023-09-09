#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:50:19 2023

@author: thomasbaycroft
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import corner
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from PyPDF2 import PdfMerger
# import pykima
from . import make_plots as mplt
from . import orbit_plot as op
from . import functions as fcs
import os
from astropy import constants
Ms = constants.M_sun.value
Mj = constants.M_jup.value

colours = ['r','b','g','m','c','darkviolet','orange']

def make_binary_sheet(system,outputloc,posteriors, len_posts, indices,indices_bin,full_posteriors,numpl,BFs_nps,MaxLiSolution,pri_data_list,sec_data_list=[],binary=False,SB2=False,full_sample=None,precession=False):
    
    fig = plt.figure(constrained_layout=True)
    
    gs = GridSpec(9,5, figure=fig)
    
    #binary phased rv plot
    gs1 = gs[1:4,:3].subgridspec(3,1,hspace=0)
    #np histogram
    gs2 = gs[4:6,:3].subgridspec(1,1)
    #binary parameters text and bisector plot
    gs3 = gs[1:6,3:].subgridspec(2,1)
    #periodogram, and extra parameters
    gs4 = gs[6:9,:].subgridspec(3,4, wspace=0.)
    
    #blank at the top to write text
    ax1 = plt.subplot(gs[0,:], frameon=False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.tick_params(labelbottom=False, labelleft=False)
    
    write_header_text(ax1, numpl, pri_data_list, sec_data_list=sec_data_list,SB2=SB2)
    
    ax2 = plt.subplot(gs1[0:2,:])
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.tick_params(direction="in")
    
    ax3 = plt.subplot(gs1[2,:])
    ax3.grid(True)
    ax3.set(xlabel='phase')
    ax3.tick_params(direction="in")
    
    
    for i,data in enumerate(pri_data_list):
        if binary:
            mplt.plot_rv_orbit(ax2, orb=data.binary_orbit, data=data, body='binary',colour=colours[i])
            mplt.plot_rv_orbit(ax3, orb=data.binary_orbit, data=data, body='residuals',colour=colours[i])
        if SB2:
            data = sec_data_list[i]
            mplt.plot_rv_orbit(ax2, orb=data.binary_orbit, data=data, body='binary',fill=False,colour=colours[i])
            mplt.plot_rv_orbit(ax3, orb=data.binary_orbit, data=data, body='residuals',fill=False,colour=colours[i])
    
    BFs = BFs_nps[0]
    Nps = BFs_nps[1]
    BFsamps = BFs_nps[2]
    maxNp = int(posteriors['maxNp'][0])
    
    ax5 = plt.subplot(gs2[0,0])
    #setup the bins for the Np histogram, centered on integers
    Bins = np.linspace(0, maxNp, maxNp + 1)
    #plot the np histogram
    ax5.bar(Bins,Nps)
    #identify bar correspndong to the number of planets found and make it red
    ax5.patches[numpl].set_facecolor('r')
    #write the Bayes factor for the correct bar as well as for the one above
    for i in range(maxNp):
        ax5.text(i+1,Nps[i+1],str(round(BFs[i],1)), ha='center')
        ax5.text(i+1,Nps[i+1],str(round(BFsamps[i],1)), ha='center',va='top',alpha=0.7)
    ax5.set(xlabel='Np', ylabel='Frequency')
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax6 = plt.subplot(gs4[0:2,:])
    
    #get dataspan
    tmins,tmaxs = [], []
    for data in pri_data_list:
        tmins.append(min(data.bjds))
        tmaxs.append(max(data.bjds))
    dataspan = max(tmaxs)-min(tmins)
    
    #set up the period bins
    numbins = 50
    Pbins = np.logspace(np.log10(min(full_posteriors['P'])),np.log10(max(full_posteriors['P'])),num=numbins+1) 
    #plot the periodogram    
    ax6.hist(full_posteriors['P'],bins=Pbins)
    ax6.set(xlabel='Period[days]', ylabel='Frequency')
    ax6.semilogx()
    ax6.grid(True)
    ax6.axvline(365,c='r', label='1 year')
    ax6.axvline(dataspan,c='g', label='dataspan')
    toplim = ax6.axes.get_ylim()[1]
    ax6.text(365,-toplim/9,'365',ha='center',c='r')
    ax6.text(dataspan,-toplim/9,str(round(dataspan)),ha='center',c='g')
    
    ax7 = plt.subplot(gs3[0,0], frameon=False)
    ax7.axes.get_yaxis().set_visible(False)
    ax7.axes.get_xaxis().set_visible(False)
    ax7.tick_params(labelbottom=False, labelleft=False)
    
    if binary:
        MaxLiSolution['a_bin'] = MaxLiSolution['a_pri']+MaxLiSolution['a_sec']
        parameter_names = [r'$P_{obs}$ = ',r'$K$ = ',r'$e$ = ',r'$\omega$ = ',r'$M_{pri}\sin(i)$ = ',r'$M_{sec}\sin(i)$ = ',r'$a_{bin}$',r'$T_{per}$ = ',r'$\dot{\omega}$ = ',r'$P_{ano}$ = ',r'$V_{sys} = $']
        parameter_cols = ['KO_P0','KO_K0','KO_ecc0','KO_w0','M_pri','M_sec','a_bin','Tper_bin','KO_wdot0','KO_Pano0','vsys']
        parameter_units = [' days',' m/s',' ',' rad',r' $M_{\odot}$',r' $M_{\odot}$','AU',' ',' arcsec/yr',' days','m/s']
        if SB2:
            parameter_names.append(r'$q$ = ')
            parameter_cols.append('KO_q0')
            parameter_units.append(' ')
            parameter_names.append(r'$V_{sys,2}$ = ')
            parameter_cols.append('vsys_sec')
            parameter_units.append('m/s')
        
        texty = 'Maximum Likelihood Parameters:\n'
        for i,col in enumerate(parameter_cols):
            texty+= parameter_names[i]+str(float(MaxLiSolution[col]))+parameter_units[i]+'\n'
        ax7.text(0.,0.,texty)
    
    ax8 = plt.subplot(gs3[1,0])
    ax8.grid(True)
    ax8.tick_params(direction="in")
    for i,data in enumerate(pri_data_list):
        if data.bisfw:
            if binary:
                res = data.remove_orbit_from_data(data.bjds,data.vrad,data.binary_orbit) - data.vsys
            else:
                res = data.vrad - data.vsys
            ax8.errorbar(res,data.bis,xerr=data.svrad,yerr=2*data.svrad,c='grey',fmt='.',alpha=0.6)
            ax8.errorbar(res,data.bis,c=colours[i],fmt='.')
        ax8.set(xlabel='Res (m/s)',ylabel='Bis_span (m/s)')
        
    
    ax9 = plt.subplot(gs4[2,:], frameon=False)
    ax9.axes.get_yaxis().set_visible(False)
    ax9.axes.get_xaxis().set_visible(False)
    ax9.tick_params(labelbottom=False, labelleft=False)
    write_footer_text(ax9,posteriors,len_posts,numpl,MaxLiSolution,pri_data_list,sec_data_list)
    
    fig.set_size_inches(8.25,11.75)
    fig.suptitle(system)   
    plt.savefig(outputloc+system+'_1_datasheet.pdf')
    
    pdf_merger = PdfMerger()
    pdf_merger.append(outputloc+system+'_1_datasheet.pdf')
    
    if binary:
        cols = ['KO_P0','KO_K0','KO_ecc0','KO_w0','KO_phi0','vsys']
        names = [r'$P$',r'$K$',r'$e$',r'$\omega$',r'$\phi_0$',r'$V_{sys}$']
        units = [' days',' m/s','',' rad',' rad',' m/s']
        if precession:
            cols.append('KO_wdot0')
            names.append(r'$\dot{\omega}$')
            units.append(' arcsec/yr')
        if SB2:
            cols.append('KO_q0')
            names.append(r'$q$ ')
            units.append('')
        fig = plt.figure(tight_layout=True)
        
        ind=indices_bin
        # for i in range(1,len(indices)):
        #     ind = np.logical_or(ind,indices[i])
        ax1 = corner.corner(posteriors[cols][ind],show_titles=True,title_fmt='g',quantiles=[0.16,0.5,0.84],labels=[names[i]+units[i] for i in range(len(names))],titles=names)
        fig.set_size_inches(8.25,8.25)
        plt.savefig(outputloc+'corner.pdf')
   
        
        pdf_merger.append(outputloc+'corner.pdf')
        
    # with open('datasheet.pdf',mode='wb') as output_file:
        
    pdf_merger.write(outputloc+system+'_1_datasheet.pdf') 
    
    os.system('rm '+outputloc+'corner.pdf')
    
def make_object_sheet(system,outputloc,Known_object,object_number,sheet_number,posteriors,indices,full_posteriors,numpl,BFs_nps,MaxLiSolution,pri_data_list,sec_data_list=[],binary=False,SB2=False,full_sample=None,precession=False):
    
    if Known_object:
        prefix = 'KO_'
    else:
        prefix = ''
    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure()
    
    gs = GridSpec(5,5, figure=fig)
    
    #binary phased rv plot
    gs1 = gs[1:5,:3].subgridspec(3,1)#,hspace=0)
    #binary parameters text and bisector plot
    gs2 = gs[1:5,3:].subgridspec(3,1)
    
    #blank at the top to write text
    ax1 = plt.subplot(gs[0,:], frameon=False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.tick_params(labelbottom=False, labelleft=False)
    
    write_header_text(ax1, numpl, pri_data_list, sec_data_list=sec_data_list,SB2=SB2)
    
    ax2 = plt.subplot(gs1[0:2,:])
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.tick_params(direction="in")
    
    ax3 = plt.subplot(gs1[2,:])
    ax3.grid(True)
    ax3.set(xlabel='phase')
    ax3.tick_params(direction="in")
    
    
    for i,data in enumerate(pri_data_list):
        if Known_object:
            orbit = data.KO_orbits[object_number]
            mplt.plot_rv_orbit(ax2, orb=orbit, data=data, body='',colour=colours[i])
            mplt.plot_rv_orbit(ax3, orb=orbit, data=data, body='residuals',colour=colours[i])
            if SB2:
                data = sec_data_list[i]
                orbit = data.KO_orbits[object_number]
                mplt.plot_rv_orbit(ax2, orb=orbit, data=data, body='',fill=False,colour=colours[i])
                mplt.plot_rv_orbit(ax3, orb=orbit, data=data, body='residuals',fill=False,colour=colours[i])
        else:
            orbit = data.planet_orbits[object_number]
            mplt.plot_rv_orbit(ax2, orb=orbit, data=data, body='',colour=colours[i])
            mplt.plot_rv_orbit(ax3, orb=orbit, data=data, body='residuals',colour=colours[i])
            if SB2:
                data = sec_data_list[i]
                orbit = data.planet_orbits[object_number]
                mplt.plot_rv_orbit(ax2, orb=orbit, data=data, body='',fill=False,colour=colours[i])
                mplt.plot_rv_orbit(ax3, orb=orbit, data=data, body='residuals',fill=False,colour=colours[i])
    
    ax7 = plt.subplot(gs2[0:2,0], frameon=False)
    ax7.axes.get_yaxis().set_visible(False)
    ax7.axes.get_xaxis().set_visible(False)
    ax7.tick_params(labelbottom=False, labelleft=False)
    
    s = str(object_number)
    parameter_names = [r'$P$ = ',r'$K$ = ',r'$e$ = ',r'$\omega$ = ',r'$m\sin(i)$ = ',r'$a$ = ',r'$T_{per}$ = ',r'$V_{sys} = $']
    if float(MaxLiSolution['M_pl_'+s]) >= 0.08:
        parameter_cols = ['P'+s,'K'+s,'ecc'+s,'w'+s,'M_pl_'+s,'a_pl_'+s,'Tper'+s,'vsys']
        parameter_units = [' days',' m/s',' ',' rad',r' $M_{\odot}$',' AU',' ',' days','m/s']
    else:
        MaxLiSolution['M_pl_j_'+s] = MaxLiSolution['M_pl_'+s]*Ms/Mj
        parameter_cols = ['P'+s,'K'+s,'ecc'+s,'w'+s,'M_pl_j_'+s,'a_pl_'+s,'Tper'+s,'vsys']
        parameter_units = [' days',' m/s',' ',' rad',r' $M_{\rm Jup}$',' AU',' ',' days','m/s']
    if precession:
        parameter_cols.append('wdot'+s)
        parameter_names.append(r'$\dot{\omega}$ = ')
        parameter_units.append(' arcsec/yr')
    
    
    texty = 'Maximum Likelihood Parameters:\n'
    for i,col in enumerate(parameter_cols):
        texty+= parameter_names[i]+str(float(MaxLiSolution[col]))+parameter_units[i]+'\n'
    ax7.text(0.,0.,texty)
    
    
    ax8 = plt.subplot(gs2[2,0])
    ax8.grid(True)
    ax8.tick_params(direction="in")
    for i,data in enumerate(pri_data_list):
        if data.bisfw:
            if binary:
                res = data.remove_orbit_from_data(data.bjds,data.vrad,data.binary_orbit) - data.vsys
                if Known_object:
                    res = data.remove_orbit_from_data(data.bjds,res,data.KO_orbits[object_number])
                else:
                    res = data.remove_orbit_from_data(data.bjds,res,data.planet_orbits[object_number])
            else:
                if Known_object:
                    res = data.remove_orbit_from_data(data.bjds,data.vrad,data.KO_orbits[object_number]) - data.vsys
                else:
                    res = data.remove_orbit_from_data(data.bjds,data.vrad,data.planet_orbits[object_number]) - data.vsys
            ax8.errorbar(res,data.bis,xerr=data.svrad,yerr=2*data.svrad,c='grey',fmt='.',alpha=0.6)
            ax8.errorbar(res,data.bis,c=colours[i],fmt='.')
        ax8.set(xlabel='Res (m/s)',ylabel='Bis_span (m/s)')
    
    fig.set_size_inches(10,12)
    fig.suptitle(system)
    if Known_object:
        loc = outputloc+system+'_'+str(sheet_number)+'_'+'known_object'+str(object_number+1)
    else:
        loc = outputloc+system+'_'+str(sheet_number)+'_'+'planet'+str(object_number+1)
    
    plt.savefig(loc+'_datasheet.pdf')
    
    pdf_merger = PdfMerger()
    pdf_merger.append(loc+'_datasheet.pdf')
    
    if Known_object:
        s = str(object_number+1)
        cols = ['KO_P'+s,'KO_K'+s,'KO_ecc'+s,'KO_w'+s,'KO_phi'+s,'KO_vsys']
        names = [r'$P$',r'$K$',r'$e$',r'$\omega$',r'$\phi_0$',r'$V_{sys}$']
        units = [' days',' m/s','',' rad',' rad',' m/s']
        if precession:
            parameter_cols.append('KO_wdot'+s)
            parameter_names.append(r'$\dot{\omega}$ = ')
            parameter_units.append(' arcsec/yr')
        ind=indices[0]
        for i in range(1,len(indices)):
            ind = np.logical_or(ind,indices(i))
    else:
        cols = ['P','K','ecc','w','phi','vsys']
        names = [r'$P$',r'$K$',r'$e$',r'$\omega$',r'$\phi_0$',r'$V_{sys}$']
        units = [' days',' m/s','',' rad',' rad',' m/s']
        if precession:
            parameter_cols.append('wdot')
            parameter_names.append(r'$\dot{\omega}$ = ')
            parameter_units.append(' arcsec/yr')
        ind = indices[object_number]
        
    fig = plt.figure(tight_layout=True)

    ax1 = corner.corner(posteriors[cols][ind],show_titles=True,title_fmt='g',quantiles=[0.16,0.5,0.84],labels=[names[i]+units[i] for i in range(len(names))],titles=names)
    fig.set_size_inches(8.25,8.25)
    plt.savefig(outputloc+'corner.pdf')
    
    pdf_merger.append(outputloc+'corner.pdf')
    
    pdf_merger.write(loc+'_datasheet.pdf') 
    
    os.system('rm '+outputloc+'corner.pdf')
    
def make_timeseries_sheet(system,outputloc,sheet_number,instruments,MaxLiSolution,numpl,nKOs,pri_data_list,sec_data_list=[],binary=False,SB2=False,P_prior=[1.0,2000],drift_order=0):
    
    l = len(pri_data_list) + bool(drift_order>=1)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(l,1,figure=fig)
    times=[]
    for i,data in enumerate(pri_data_list):
        ax = plt.subplot(gs[i,0])
        ax.errorbar(data.bjds,data.residuals(),yerr=data.svrad,fmt='.',c='b',ecolor='darkgrey')
        if SB2:
            dat = sec_data_list[i]
            ax.errorbar(dat.bjds,dat.residuals(),yerr=dat.svrad,fmt='.',c='b',mfc='none',ecolor='darkgrey')
        ax.set(xlabel='date (bjd)',ylabel='Res (m/s)')
        leftlim = ax.axes.get_xlim()[0]
        toplim = ax.axes.get_ylim()[1]
        rightlim = ax.axes.get_xlim()[1]
        ax.text((leftlim+rightlim)/2,toplim,str(instruments[i]),ha='center',va='top',c='m')
        ax.axhline(0,c='black',alpha=0.6)
        times += list(data.bjds)
        
    if drift_order>=1:
        ax = plt.subplot(gs[i+1,0])
        for i,data in enumerate(pri_data_list):
            ax.errorbar(data.bjds,data.residuals()+data.drift(),yerr=data.svrad,fmt='.',c='b',ecolor='darkgrey')
            if SB2:
                dat = sec_data_list[i]
                ax.errorbar(dat.bjds,dat.residuals()+data.drift(),yerr=dat.svrad,fmt='.',c='b',mfc='none',ecolor='darkgrey')
        cont = np.arange(min(times),max(times),0.1)
        ax.plot(cont,pri_data_list[0].drift(times=cont),c='black',alpha=0.8)
    
    fig.set_size_inches(7,4)
    plt.savefig(outputloc+system+'_'+str(sheet_number)+'_timeseries.pdf')
    
    fig,ax2 = plt.subplots(1,subplot_kw={'projection': 'polar'})
    Tpers,pl_Ps,pl_Ms,pl_es,pl_ws = [], [], [], [], []

    for i in range(1,nKOs):
        pl_Ps.append(float(MaxLiSolution['KO_P'+str(i)]))
        pl_Ms.append(float(MaxLiSolution['M_KO_'+str(i)]))
        pl_es.append(float(MaxLiSolution['KO_ecc'+str(i)]))
        pl_ws.append(float(MaxLiSolution['KO_w'+str(i)]))
        Tpers.append(float(MaxLiSolution['KO_Tper'+str(i)]))
    for i in range(numpl):
        pl_Ps.append(float(MaxLiSolution['P'+str(i)]))
        pl_Ms.append(float(MaxLiSolution['M_pl_'+str(i)]))
        pl_es.append(float(MaxLiSolution['ecc'+str(i)]))
        pl_ws.append(float(MaxLiSolution['w'+str(i)]))
        Tpers.append(float(MaxLiSolution['Tper'+str(i)]))
    
    plot = False
    if binary:
        plot = True
        Bin_P,M1,M2,Bin_e,Bin_w = float(MaxLiSolution['KO_P0']),float(MaxLiSolution['M_pri']),float(MaxLiSolution['M_sec']),float(MaxLiSolution['KO_ecc0']),float(MaxLiSolution['KO_w0'])
        Tpers = [float(MaxLiSolution['Tper_bin']),float(MaxLiSolution['Tper_bin'])]+Tpers
        op.plot_circumbinary(Bin_P, M1, M2, Bin_e, Bin_w, pl_Ps, pl_Ms, pl_es, pl_ws,timeseries=times,tpers=Tpers,SB2=SB2,ax=ax2,fig=fig)
    else:
        if numpl >= 1 or nKOs >=1:
            plot = True
            op.plot_orbits(np.array(pl_Ms), np.array(pl_Ps), np.array(pl_es), np.array(pl_ws),timeseries=np.array(times),tpers=np.array(Tpers),ax=ax2,fig=fig)
    
    if plot:
        fig.set_size_inches(5,5)
        plt.savefig(outputloc+'orbit.pdf')
        
        pdf_merger = PdfMerger()
        pdf_merger.append(outputloc+system+'_'+str(sheet_number)+'_timeseries.pdf')
        
        pdf_merger.append(outputloc+'orbit.pdf')
        
        
        pdf_merger.write(outputloc+system+'_'+str(sheet_number)+'_timeseries.pdf') 
    
        os.system('rm '+outputloc+'orbit.pdf')
    
def make_extras_sheet(system,outputloc,sheet_number,full_posteriors,indices,MaxLiSolution,fiptip,times,primaries,secondaries,binary=False,P_prior=[],SB2=False):
    
    
    # fig = plt.figure(constrained_layout=True)
    fig = plt.figure()
    gs = GridSpec(5,1,figure=fig)
    
    gs1 = gs[0,:].subgridspec(1,2)
    ax1 = plt.subplot(gs1[0,0])
    ax2 = plt.subplot(gs1[0,1])
    
    pl_ind=np.zeros(len(full_posteriors),dtype=int)
    for ind in indices:
        pl_ind = np.logical_or(ind,pl_ind)
    no_pl_ind = np.logical_not(pl_ind)
    
    
    posts_C = full_posteriors[no_pl_ind]
    
    M_central = float(MaxLiSolution.M_pri)
    if binary:
        M_central+=float(MaxLiSolution.M_sec)
    
    mplt.plot_detection_limit(ax1, posts_C,M_central=M_central,KM='K')
    
    ax2.scatter(full_posteriors.P,full_posteriors.logL,c='b',alpha=0.1,s=1)
    if binary:
        ax2.scatter(full_posteriors.KO_P0,full_posteriors.logL,c='b',alpha=0.1,s=1)
    ax2.semilogx()
    ax2.set(ylim=[np.median(full_posteriors.logL),max(full_posteriors.logL)+0.5],xlabel='Period (days)',ylabel=r'$log(L)$')
    
    gs2 = gs[1,:].subgridspec(1,3)
    ax3 = plt.subplot(gs2[0,0])
    ax4 = plt.subplot(gs2[0,1])
    ax5 = plt.subplot(gs2[0,2])
    
    colrs = ['r','g','m','c','darkviolet','orange']
    
    ax3.scatter(full_posteriors.P[no_pl_ind],full_posteriors.ecc[no_pl_ind],c='b',alpha=0.1,s=1)
    for i,ind in enumerate(indices):
        ax3.scatter(full_posteriors.P[ind],full_posteriors.ecc[ind],c=colrs[i],alpha=0.1,s=1)
    ax3.set(xlabel='Period (days)',ylabel='Eccentricity')
    ax3.semilogx()
    
    ax4.scatter(full_posteriors.ecc[no_pl_ind],full_posteriors.K[no_pl_ind],c='b',alpha=0.1,s=1)
    for i,ind in enumerate(indices):
        ax4.scatter(full_posteriors.ecc[ind],full_posteriors.K[ind],c=colrs[i],alpha=0.1,s=1)
    ax4.set(xlabel='Eccentricity',ylabel='Semi-Amplitude (m/s)')
    ax4.semilogy()

    ax5.scatter(full_posteriors.P[no_pl_ind],full_posteriors.K[no_pl_ind],c='b',alpha=0.1,s=1)
    for i,ind in enumerate(indices):
        ax5.scatter(full_posteriors.P[ind],full_posteriors.K[ind],c=colrs[i],alpha=0.1,s=1)
    ax5.set(xlabel='Period (days)',ylabel='Semi-Amplitude (m/s)')
    ax5.semilogx()
    ax5.semilogy()
        
    gs3 = gs[2,:].subgridspec(1,2)
    ax6 = plt.subplot(gs3[0,0])
    ax7 = plt.subplot(gs3[0,1])
    
    ax6.plot(fiptip[0],np.log10(fiptip[1]),c='b',label='Posts')
    ax6.plot(fiptip[0],np.log10(fiptip[2]),c='r',label='Samps')
    ax6.set(xlabel='Period (days)',ylabel=r'$log_{10}$ FIP',ylim=[-3,0.1])
    ax6.axhline(np.log10(0.01))
    ax6.semilogx()
    ax6.legend()
    
    ax7.plot(fiptip[0],np.log10(1-fiptip[1]),c='b',label='Posts')
    ax7.plot(fiptip[0],np.log10(1-fiptip[2]),c='r',label='Samps')
    ax7.set(xlabel='Period (days)',ylabel=r'$log_{10}$ TIP')
    ax7.axhline(np.log10(0.99))
    ax7.semilogx()
    ax7.legend()
    
    tiper = 1-fiptip[2]
    if max(tiper) >= 0.2:
        ax7.text(fiptip[0][np.argmax(tiper)],np.log10(max(tiper)),str(round(max(tiper),2)),ha='center')
    # if max(1-fiptip[1]) >= 0.2:
    #     axs[1,1].text(pers[np.argmax(tiper_posts)],np.log10(max(tiper_posts)),str(round(max(tiper_posts),2)),ha='center')
      
    ax8 = plt.subplot(gs[3,0])
    if P_prior == []:
        f1,f2 = 1/2000,1/1
    else:
        f1,f2 = 1/P_prior[1],1/P_prior[0]
    freqs = np.linspace(f1,f2,len(times)*18)
    wfpower = fcs.get_spectral_window_function(freqs, times)
    ax8.plot(1/freqs,wfpower,c='b')
    ax8.semilogx()
    ax8.set(xlabel='Period (days)',ylabel='Window function')
    
    gs4 = gs[4,:].subgridspec(1,2)
    if SB2:
        times,rvs,errs=[],[],[]
        for pri in primaries:
            times += list(pri.bjds)
            rvs += list(pri.rvs_no_binary())
            errs += list(pri.svrad)
        freqs,power,faplevels,plevels,maxfreq = fcs.GLS_periodogram(times, rvs, errs,p_range=[1/f2,1/f1])
        ax9 = plt.subplot(gs4[0])
        ax9.plot(1/freqs,power)
        ax9.semilogx()
        for i,lev in enumerate(plevels):
            ax9.axhline(lev,ls='--',c='grey',alpha=0.8)
            ax9.text(max(1/freqs),lev,str(faplevels[i]),ha='right')
        ax9.axvline(1/maxfreq,c='g')
        ax9.axvline(365.25,c='r',ls=':')
        ax9.set(xlabel='Period (days)',ylabel='power',title='Primary')
        
        times,rvs,errs=[],[],[]
        for sec in secondaries:
            times += list(sec.bjds)
            rvs += list(sec.rvs_no_binary())
            errs += list(sec.svrad)
        freqs,power,faplevels,plevels,maxfreq = fcs.GLS_periodogram(times, rvs, errs,p_range=[1/f2,1/f1])
        ax10 = plt.subplot(gs4[1])
        ax10.plot(1/freqs,power)
        ax10.semilogx()
        for i,lev in enumerate(plevels):
            ax10.axhline(lev,ls='--',c='grey',alpha=0.8)
            ax10.text(max(1/freqs),lev,str(faplevels[i]),ha='right')
        ax10.axvline(1/maxfreq,c='g')
        ax10.axvline(365.25,c='r',ls=':')
        ax10.set(xlabel='Period (days)',ylabel='power',title='Secondary')
    else:
        times,rvs,errs=[],[],[]
        for pri in primaries:
            times += list(pri.bjds)
            rvs += list(pri.rvs_no_binary())
            errs += list(pri.svrad)
        freqs,power,faplevels,plevels,maxfreq = fcs.GLS_periodogram(times, rvs, errs,p_range=[1/f2,1/f1])
        ax9 = plt.subplot(gs4[:])
        ax9.plot(1/freqs,power)
        ax9.semilogx()
        for i,lev in enumerate(plevels):
            ax9.axhline(lev,ls='--',c='grey',alpha=0.8)
            ax9.text(max(1/freqs),lev,str(faplevels[i]),ha='right')
        ax9.axvline(1/maxfreq,c='g')
        ax9.axvline(365.25,c='r',ls=':')
        ax9.set(xlabel='Period (days)',ylabel='power')
            
    
    fig.set_size_inches(10.25,18.75)
    plt.savefig(outputloc+system+'_'+str(sheet_number)+'_extras.png')
    
    

def make_diagnostics(system,outputloc,sheet_number,loc):
    fig = plt.figure(constrained_layout=True)
    gs2 = GridSpec(2,2,figure=fig)
    
    #gs2 = gs[3:,:].subgridspec(2,2)
    ax1 = plt.subplot(gs2[:,0])
    ax2 = plt.subplot(gs2[0,1])
    ax3 = plt.subplot(gs2[1,1])
    
    with open(loc+'sample_info.txt') as f:
        names = f.readline()[1:].split(',')
    Sampleinfo = pd.read_csv(loc+'sample_info.txt',header=None,skiprows=[0],names=names,sep='\s+',engine='python')
    
    with open(loc+'levels.txt') as f:
        names = f.readline()[1:].split(',')
    Levels = pd.read_csv(loc+'levels.txt',header=None,skiprows=[0],names=names,sep='\s+',engine='python')
    
    num2=len(Sampleinfo)
    iteration = np.linspace(0,num2-1,num2)
    ax1.plot(iteration,Sampleinfo[' level assignment'])
    ax1.set(ylabel='Level', xlabel='Iteration')

    num3=len(Levels)
    lev = np.linspace(0,num3-1,num3)
    ax2.plot(lev,Levels[' accepts']/Levels[' tries'])
    ax2.set(ylim=[0,1],ylabel='MH Acceptance')
    ax2.axhline(0.2,c='r',linestyle='dotted')
    

    lev = np.linspace(0,num3-2,num3-1)
    ax3.plot(lev,np.array(Levels[' log_X'][1:])-np.array(Levels[' log_X'][:-1]))
    ax3.set(ylim=[-2.5,0],ylabel='Compression',xlabel='Level')
    ax3.axhline(-1,c='r',linestyle='dotted')

    
    fig.set_size_inches(8.25,4)
    plt.savefig(outputloc+system+'_'+str(sheet_number)+'_diagnostics1.pdf')
    
    # plt.close('all')
    # matplotlib.use('Agg')
    # pykima.postprocess()
    # plt.figure(3)
    # plt.savefig(outputloc+'diagnostic_2.pdf')
    
    # pdf_merger = PdfMerger()
    
    # pdf_merger.append(outputloc+system+'_diagnostics1.pdf')
    # pdf_merger.append(outputloc+'diagnostics2.pdf')

    # pdf_merger.write(outputloc+system+'_'+str(sheet_number)+'_diagnostics.pdf')   
      
    # os.system('rm '+outputloc+system+'_'+str(sheet_number)+'_diagnostics1.pdf'+' '+outputloc+'diagnostics2.pdf')

    
def write_header_text(ax,numpl,pri_data_list,sec_data_list=[],SB2=False):

    for i,data in enumerate(pri_data_list):
        if SB2:
            data2 = sec_data_list[i]
            errs = str(round(data.mean_rv_err(),2))+'/'+str(round(data2.mean_rv_err(),2))
            rms = str(round(data.rms(),2))+'/'+str(round(data2.rms(),2))
            jit = str(round(data.jitter,2))+'/'+str(round(data2.jitter,2))
            texty = 'Mean RVerrs = '+errs+' m/s\t'
            texty += 'RMS = '+rms+' m/s\t'
            texty += 'Jits = '+jit+' m/s'
        else:
            errs = str(round(data.mean_rv_err(),2))
            rms = str(round(data.rms(),2))
            jit = str(round(data.jitter,2))
            texty = 'Mean RVerr = '+errs+' m/s\t'
            texty += 'RMS = '+rms+' m/s\t'
            texty += 'Jit = '+jit+' m/s'
        ax.text(0.,float(i/len(pri_data_list)),data.inst_name+r': # data $=$ '+str(len(data.vrad))+'\t'\
                 + texty, c=colours[i])
            
    numdat=0
    datsum=0
    chi_squared_red = 1.0
    if SB2:
        for i,data in enumerate(pri_data_list):
            numdat += len(data.bjds)
            datsum+=np.sum(data.bjds)
            
    else:
        for i,data in enumerate(pri_data_list):
            numdat += len(data.bjds)
            datsum+=np.sum(data.bjds)
        
    ax.text(0.,1.,'# data = '+str(numdat)+'\t\t'\
             + r'$\chi^2_{\nu}$ = '+str(round(chi_squared_red,2))+'\t'+'T0 = '+str(round(datsum/numdat,6)))
    
def write_footer_text(ax,posteriors,len_posts,numpl,MaxLiSample,pri_data_list,sec_data_list=[],SB2=False):
    texty1=''
    texty1+='# samples = '+str(len_posts) +'\n'
    texty1+='# Np samples = '+str(int(len(posteriors)/max(numpl,1))) +'\n'
    try:
        nu = str(round(float(MaxLiSample.nu),1))
        texty1 += 'nu = '+nu+'\n'
    except:
        pass
    dm=[]
    dM=[]
    for data in pri_data_list:
        dm.append(min(data.bjds))
        dM.append(max(data.bjds))
    texty1+='dataspan = '+ str(round(max(dM)-min(dm),1))+' days\n'
    
    ax.text(0.,0.,texty1)
    
    # texty2=''
    
    # ax.text(0.5,0.,Tid_effect_size+GR_effect_size+M_central)
    pass    
