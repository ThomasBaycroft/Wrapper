import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import corner
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import os
from PyPDF2 import PdfMerger
from kima.pykima.classic import postprocess
from PIL import Image

from . import functions as fcs
from . import GAIAdata_analysis as gda
from . import make_plots as mplt

colours = ['cornflowerblue','indianred','forestgreen','magenta','darkviolet','orange','b','r','g','m','c','black','black','black','black','black','black','black','black','black']


def make_baseline_sheet(system, outputloc, posteriors, len_posts, indices, indices_full, posteriors_np, posteriors_single, full_posteriors, nKOs, numpl, BFs_nps, fiptip, res, pykima_ML, MaxLiSolution, ML_param_names, ML_param_cols, ML_param_units, value_mult, dataspan, noise_params, noise_params_names, acceleration, jerk):

    fig = plt.figure(constrained_layout=True)
    
    gs = GridSpec(6,5, figure=fig)
    
    #Np histogram
    gs1 = gs[:2,:3].subgridspec(1,1,hspace=0)
    #period histogram
    gs2 = gs[2:4,:3].subgridspec(1,1)
    #FIP
    gs3 = gs[4:6,:3].subgridspec(1,1)
    #Maxli Params
    gs4 = gs[:4,3:].subgridspec(1,1, wspace=0.)
    #Noise Params
    gs5 = gs[4:6,3:].subgridspec(1,1, wspace=0.)

    #Maxli Params box
    ax4 = plt.subplot(gs4[:,:], frameon=False)
    ax4.axes.get_yaxis().set_visible(False)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.tick_params(labelbottom=False, labelleft=False)

    texty = 'Maximum Likelihood Parameters:\n'
    for i,col in enumerate(ML_param_cols):
        texty+= ML_param_names[i]+ML_param_units[i]+' = '+str(round(float(MaxLiSolution[col])*value_mult[i],6))+'\n'
    ax4.text(0.,1.,texty,va='top',fontsize='large')

    #NoiseParams box
    ax5 = plt.subplot(gs5[:,:], frameon=False)
    ax5.axes.get_yaxis().set_visible(False)
    ax5.axes.get_xaxis().set_visible(False)
    ax5.tick_params(labelbottom=False, labelleft=False)

    texty2 = 'Data, Fit, and Noise Parameters:\n'
    for i,param in enumerate(noise_params):
        texty2+= noise_params_names[i]+str(param)+'\n'
    ax5.text(0.,1.,texty2,va='top',fontsize='large')

    BFs = BFs_nps[0]
    Nps = BFs_nps[1]
    BFsamps = BFs_nps[2]
    maxNp = int(posteriors['maxNp'][0])
    
    ax1 = plt.subplot(gs[:2,:3])
    #setup the bins for the Np histogram, centered on integers
    Bins = np.linspace(0, maxNp, maxNp + 1)
    #plot the np histogram
    ax1.bar(Bins,Nps)
    #identify bar correspndong to the number of planets found and make it red
    ax1.patches[numpl].set_facecolor('r')
    #write the Bayes factor for the correct bar as well as for the one above
    for i in range(maxNp):
        ax1.text(i+1,Nps[i+1],str(round(BFs[i],1)), ha='center')
        ax1.text(i+1,Nps[i+1],str(round(BFsamps[i],1)), ha='center',va='top',alpha=0.7)
    ax1.set(xlabel='Np', ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    #Periodogram boxes
    ax2 = plt.subplot(gs2[:,:])
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.tick_params(direction="in")
    
    ax3 = plt.subplot(gs3[:,:])
    ax3.grid(True)
    ax3.set(xlabel='phase')
    ax3.tick_params(direction="in")

    #set up the period bins
    numbins = 50
    Pbins = np.logspace(np.log10(min(full_posteriors['P'])),np.log10(max(full_posteriors['P'])),num=numbins+1)
    #plot the periodogram    
    ax2.hist(full_posteriors['P'],bins=Pbins)
    ax2.set(xlabel='Period[days]', ylabel='Frequency')
    ax2.semilogx()
    ax2.grid(True)
    ax2.axvline(365,c='r', label='1 year')
    ax2.axvline(dataspan,c='g', label='dataspan')
    toplim = ax2.axes.get_ylim()[1]
    ax2.text(365,-toplim/9,'365',ha='center',c='r')
    ax2.text(dataspan,-toplim/9,str(round(dataspan)),ha='center',c='g')

    ax3.plot(fiptip[0],np.log10(fiptip[1]),c='b',label='Posts')
    ax3.plot(fiptip[0],np.log10(fiptip[2]),c='r',label='Samps')
    ax3.set(xlabel='Period (days)',ylabel=r'$log_{10}$ FIP',ylim=[-3,0.1])
    ax3.axhline(np.log10(0.01))
    ax3.semilogx()
    ax3.legend()

    fig.set_size_inches(12,10)
    plt.savefig(outputloc+system+'_'+str(1)+'_main_temp.pdf')

    pdf_merger = PdfMerger()
    rems = []
    pdf_merger.append(outputloc+system+'_'+str(1)+'_main_temp.pdf')
    rems.append(outputloc+system+'_'+str(1)+'_main_temp.pdf')

    plt.close('all')
    if res is not None:
        res.phase_plot(pykima_ML)
        plt.savefig(outputloc+system+'_'+str(1)+'_phase.pdf')
        plt.close('all')
        pdf_merger.append(outputloc+system+'_'+str(1)+'_phase.pdf')
        rems.append(outputloc+system+'_'+str(1)+'_phase.pdf')
    
    fig = plt.figure()
    fig.set_size_inches(12,12)
    column_value_low = 0
    column_value_high = 5
    if acceleration:
        column_value_high = 7
    if jerk:
        column_value_high = 9

    page_title = outputloc+system+'_'+str(1)+'_corner_baseline.pdf'

    cols = ML_param_cols[column_value_low:column_value_high]
    titles = ML_param_names[column_value_low:column_value_high]
    units = ML_param_units[column_value_low:column_value_high]
    labels = [titles[j] + units[j] for j in range(len(titles))]

    corner.corner(posteriors[cols],fig=fig,labels=labels,titles=titles,show_titles=True,title_fmt='.8g')
    plt.savefig(page_title)
    plt.close('all')
    pdf_merger.append(page_title)
    rems.append(page_title)

    column_value_low += column_value_high
    column_value_high += 7

    for i in range(nKOs):
        fig = plt.figure()
        fig.set_size_inches(12,12)
        page_title = outputloc+system+'_'+str(1)+'_corner_KO_'+str(i)+'.pdf'

        cols = ML_param_cols[column_value_low:column_value_high]
        titles = ML_param_names[column_value_low:column_value_high]
        units = ML_param_units[column_value_low:column_value_high]
        labels = [titles[j] + units[j] for j in range(len(titles))]

        corner.corner(posteriors[cols],fig=fig,labels=labels,titles=titles,show_titles=True,title_fmt='.8g')
        plt.savefig(page_title)
        plt.close('all')
        pdf_merger.append(page_title)
        rems.append(page_title)
        column_value_low += 11
        column_value_high += 11

    for i in range(numpl):
        fig = plt.figure()
        fig.set_size_inches(12,12)
        page_title = outputloc+system+'_'+str(1)+'_corner_pl_'+str(i)+'.pdf'

        cols1 = ML_param_cols[column_value_low:column_value_high]
        cols = [cols1[j][:-1] for j in range(len(cols1))]
        titles = ML_param_names[column_value_low:column_value_high]
        units = ML_param_units[column_value_low:column_value_high]
        labels = [titles[j] + units[j] for j in range(len(titles))]

        posts_pls = full_posteriors[indices_full[i]]
        corner.corner(posts_pls[cols],fig=fig,labels=labels,titles=titles,show_titles=True,title_fmt='.8g')
        plt.savefig(page_title)
        plt.close('all')
        pdf_merger.append(page_title)
        rems.append(page_title)
        column_value_low += 11
        column_value_high += 11

    pdf_merger.write(outputloc+system+'_'+str(1)+'_main.pdf')

    for rem in rems:
        os.system('rm '+rem)
  

def make_extras_sheet(system,outputloc,posteriors,full_posteriors,indices_full,numpl,data_kepmodel_format,nKOs,star_mass,plx,dataspan=2000,Gaia_epoch=2457936.875):

    #make PKE etc
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(12,12)

    gs = GridSpec(3,6, figure=fig)
    
    #Hexbin
    gs1 = gs[0,:3].subgridspec(1,1)
    #likelihood-period
    gs2 = gs[0,3:].subgridspec(1,1)
    #Pae
    gs3 = gs[1,:].subgridspec(1,3)
    #Other hists
    gs4 = gs[2,:].subgridspec(1,3)

    ax1 = plt.subplot(gs1[:,:])
    Mlow = fcs.get_mass_AST_scp(dataspan,np.median(data_kepmodel_format.centroid_pos_error_al),star_mass,plx)*fcs.Msun/fcs.Mj
    Mhigh = np.max(full_posteriors.M_pl_)*fcs.Msun/fcs.Mj
    mplt.plot_detection_limit(ax1,full_posteriors,KM='M',m_unit='Jupiter',uplim=np.log10(Mhigh),lowlim=np.log10(Mlow/10))

    ax2 = plt.subplot(gs2[:,:])
    j=0
    ax2.scatter(full_posteriors.P,full_posteriors.logL,c=colours[j],marker='.',rasterized=True)
    j+=1
    for i in range(nKOs):
        ax2.scatter(full_posteriors['KO_P'+str(i)],full_posteriors.logL,c=colours[j],marker='.',rasterized=True)
        j+=1
    for i in range(numpl):
        ax2.scatter(full_posteriors['P'][indices_full[i]],full_posteriors.logL[indices_full[i]],c=colours[j],marker='.',rasterized=True)
        j+=1
    ymin,ymax = ax2.get_ylim()
    ax2.set(xlabel='Period (days)',ylabel='log(L)',ylim=[(ymin+ymax)/2,ymax],xscale='log')

    ax3 = plt.subplot(gs3[0])
    ax4 = plt.subplot(gs3[1])
    ax5 = plt.subplot(gs3[2])

    j=0
    ax3.scatter(full_posteriors.P,full_posteriors.a,c=colours[j],marker='.',zorder=2,rasterized=True)
    j+=1
    for i in range(nKOs):
        ax3.scatter(full_posteriors['KO_P'+str(i)],full_posteriors['KO_a'+str(i)],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    for i in range(numpl):
        ax3.scatter(full_posteriors['P'][indices_full[i]],full_posteriors['a'][indices_full[i]],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    ymin,ymax = ax2.get_ylim()
    ylow = np.percentile(full_posteriors.a,2)
    ax3.set(xlabel='Period (days)',ylabel='a (mas)',xscale='log',yscale='log',ylim=[ylow,ymax])

    j=0
    ax4.scatter(full_posteriors.P,full_posteriors.ecc,c=colours[j],marker='.',zorder=2,rasterized=True)
    j+=1
    for i in range(nKOs):
        ax4.scatter(full_posteriors['KO_P'+str(i)],full_posteriors['KO_ecc'+str(i)],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    for i in range(numpl):
        ax4.scatter(full_posteriors['P'][indices_full[i]],full_posteriors['ecc'][indices_full[i]],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    ax4.set(xlabel='Period (days)',ylabel='e ',xscale='log')

    j=0
    ax5.scatter(full_posteriors.a,full_posteriors.ecc,c=colours[j],marker='.',zorder=2,rasterized=True)
    j+=1
    for i in range(nKOs):
        ax5.scatter(full_posteriors['KO_a'+str(i)],full_posteriors['KO_ecc'+str(i)],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    for i in range(numpl):
        ax5.scatter(full_posteriors['a'][indices_full[i]],full_posteriors['ecc'][indices_full[i]],c=colours[j],marker='.',zorder=3,rasterized=True)
        j+=1
    ax5.set(xlabel='a (mas)', ylabel='e',xscale='log',xlim=[ylow,ymax])

    ax6 = plt.subplot(gs4[0])
    ax7 = plt.subplot(gs4[1])
    ax8 = plt.subplot(gs4[2])

    ax6.hist(posteriors.jitter,bins=25,fc=colours[0])
    ax6.set(xlabel='Jitter [mas]',ylabel='Posterior distribution')

    ax7.hist(data_kepmodel_format.centroid_pos_error_al,bins=20,fc=colours[0])
    ax7.set(xlabel='Along-scan uncertainties [mas]',ylabel='Distribution')

    ax8.hist(data_kepmodel_format.scan_pos_angle,bins=20,fc=colours[0])
    ax8.set(xlabel='Scan-angle [deg]',ylabel='distribution')

    plt.savefig(outputloc+system+'_Pae.pdf')

    plt.close('all')
    # Periodogram settings
    Pmin = 0.95
    Pmax = 10_000
    nfreq = 50_000

    P, power, kmax, fap = gda.kepmodel_analysis_periodogram(data_kepmodel_format,Pmin=Pmin,Pmax=Pmax,nfreq=nfreq)

    psis = np.array(data_kepmodel_format.scan_pos_angle)
    times = np.array(data_kepmodel_format.obs_time_tcb)


    freqs = np.linspace(1/Pmax,1/Pmin,nfreq)
    time_array = 1/freqs

    power_a,power_d = fcs.gaia_astrometric_window_function(freqs,times,psis)
    power_al = fcs.get_spectral_window_function(freqs,times)

    # Plot periodograms
    fig, axs = plt.subplots(4,1,figsize=[12,12],constrained_layout=True)
    power_al_rest = power_al[time_array<dataspan]
    axs[0].plot(time_array,power_al,'k',lw=1.5,rasterized=True,label='Along-scan window function')
    axs[0].set(xlabel='Period (days)',ylabel='Normalised Power',ylim=[0, 1.4 * power_al_rest.max()],xlim=[Pmin, Pmax],xscale='log')
    axs[0].semilogx()
    axs[0].legend(loc=1)
    axs[0].axvline(365.25,ls='--',c='grey')
    axs[1].plot(time_array,power_a,'k',lw=1.5,rasterized=True,label=r'$\alpha_{\star}$ window function')
    axs[1].set(xlabel='Period (days)',ylabel='Normalised Power',ylim=[0, 1.4 * power_a.max()],xlim=[Pmin, Pmax],xscale='log')
    axs[1].semilogx()
    axs[1].legend(loc=1)
    axs[1].axvline(365.25,ls='--',c='grey')
    axs[2].plot(time_array,power_d,'k',lw=1.5,rasterized=True,label=r'$\delta$ window function')
    axs[2].set(xlabel='Period (days)',ylabel='Normalised Power',ylim=[0, 1.4 * power_d.max()],xlim=[Pmin, Pmax],xscale='log')
    axs[2].semilogx()
    axs[2].legend(loc=1)
    axs[2].axvline(365.25,ls='--',c='grey')

    axs[3].plot(P,power,'k',lw=1.5,rasterized=True)
    axs[3].plot(P[kmax], power[kmax], 'or', ms=4)
    axs[3].set(xlabel='Period (days)',ylabel='Normalised Power',ylim=[0, 1.4 * power.max()],xlim=[Pmin, Pmax],xscale='log')
    axs[3].axvline(365.25,ls='--',c='grey')
    axs[3].text(
        P[kmax],
        power[kmax] * 1.15,
        f'{P[kmax]:.3f} d',
        horizontalalignment='right'
        if P[kmax] > Pmax / 10
        else 'left'
        if P[kmax] < Pmin * 10
        else 'center',
        verticalalignment='center',
        color='r',
    )
    axs[3].text(
        0.99,
        0.97,
        'Kepmodel periodogram\n'+f'FAP = {fap:.2g}',
        transform=axs[3].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
    )
    plt.savefig(outputloc+system+'_periodograms.pdf')
    plt.close('all')

    pdf_merger = PdfMerger()
    rems = []
    pdf_merger.append(outputloc+system+'_Pae.pdf')
    rems.append(outputloc+system+'_Pae.pdf')
    pdf_merger.append(outputloc+system+'_periodograms.pdf')
    rems.append(outputloc+system+'_periodograms.pdf')

    pdf_merger.write(outputloc+system+'_'+str(2)+'_extras.pdf')
    
    for rem in rems:
        os.system('rm '+rem)

    return fap


def make_diagnostics_sheet(inputloc,outputloc,system):
    logZ,du,mm,y, (fig1, fig2, fig3) = postprocess(save=False)

    for fig in (fig1, fig2, fig3):
        fig.set_size_inches(6, 4)

    fig1.savefig("fig0.png", dpi=500)
    fig2.savefig("fig1.png", dpi=500)
    fig3.savefig("fig2.png", dpi=500)

    images = [Image.open(f"fig{i}.png").convert("RGB") for i in (0, 1, 2)]
    images[0].save(outputloc+system+'_'+str(3)+'diagnostics.pdf', save_all=True, append_images=images[1:])

    for i in range(3):
        os.system('rm fig'+str(i)+'.png')

    return logZ

