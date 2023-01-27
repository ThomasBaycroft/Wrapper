#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:50:19 2023

@author: thomasbaycroft
"""

import matplotlib.pyplot as plt
import numpy as np
import corner
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from . import make_plots as mplt

def make_binary_sheet(posteriors,numpl,BFs_nps,pri_data_list,sec_data_list=[],SB2=False):
    
    fig = plt.figure(constrained_layout=True)
    
    gs = GridSpec(11,5, figure=fig)
    
    #binary phased rv plot
    gs1 = gs[1:4,:3].subgridspec(3,1)
    #np histogram
    gs2 = gs[4:6,:3].subgridspec(1,1)
    #binary parameters text and bisector plot
    gs3 = gs[1:6,3:].subgridspec(2,1)
    #periodogram, posterior distribution histograms and extra parameters
    gs4 = gs[6:11,:].subgridspec(5,4, wspace=0.)
    
    #blank at the top to write text
    ax1 = plt.subplot(gs[0,:], frameon=False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.tick_params(labelbottom=False, labelleft=False)
    
    ax2 = plt.subplot(gs1[0:2,:])
    ax2.grid(True)
    ax2.set(xticks=[])
    
    ax3 = plt.subplot(gs1[2,:])
    ax3.grid(True)
    ax3.set(xlabel='Phase')
    
    for i,data in enumerate(pri_data_list):
        mplt.plot_rv_orbit(ax2, orb=data.binary_orbit, data=data, body='binary')
        mplt.plot_rv_orbit(ax3, orb=data.binary_orbit, data=data, body='residuals')
        if SB2:
            data = sec_data_list[i]
            mplt.plot_rv_orbit(ax2, orb=data.binary_orbit, data=data, body='binary',fill=False)
            mplt.plot_rv_orbit(ax3, orb=data.binary_orbit, data=data, body='residuals',fill=False)
    
    BFs = BFs_nps[0]
    Nps = BFs_nps[1]
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
    ax5.set(xlabel='Np', ylabel='Frequency')
    ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax6 = plt.subplot(gs4[0:2,:])
    #set up the period bins
    numbins = 50
    Pbins = np.logspace(np.log10(min(posteriors['P'])),np.log10(max(posteriors['P'])),num=numbins+1) 
    #plot the periodogram    
    ax6.hist(posteriors['P'],bins=Pbins)
    ax6.set(xlabel='Period[days]', ylabel='Frequency')
    ax6.semilogx()
    ax6.grid(True)
    ax6.axvline(365,c='r', label='1 year')
    # ax6.axvline(tmax-tmin,c='g', label='dataspan')
    toplim = ax6.axes.get_ylim()[1]
    ax6.text(365,-toplim/9,'365',ha='center',c='r')
    # ax6.text(tmax-tmin,-toplim/9,str(round(dataspan)),ha='center',c='g')
    
    fig.set_size_inches(8.25,11.75)
    plt.show()    
    
    
def make_object_sheet():
    pass
    
def make_timeseries_sheet():
    pass
    
def make_extras_sheet():
    pass

def make_parameters_csv():
    pass
    
def write_header_text(posteriors,numpl,pri_data_list,sec_data_list=[],SB2=False):
    
    
    pass