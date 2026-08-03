#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:55:01 2023

@author: thomasbaycroft
"""

import numpy as np
from scipy.stats import norm, t
from . import functions as fcs
import pandas as pd
from kepmodel.astro import AstroModel
from spleaf import term

class GAIAdata:
    '''
    Class which is given Gaia epoch astrometry data and can then manipulate it based on keplerian models
    '''
    
    def __init__(self,GAIAdata,t_ref=57388.5):
        
        self.data = GAIAdata
        self.times = self.data['bjd']
        self.w = self.data['w']
        self.w_err = self.data['sigw']
        self.psi = self.data['psi']
        self.pf = self.data['pf']
        
        self.t_ref = t_ref
        
        self.numpl = 0
        self.nKOs = 0
        self.num_bodies = 0
        self.planet_orbits = []
        self.KO_orbits = []
        self.jitter = 0.
        self.num_params = 0
        self.studT = False
        self.inliers = np.ones(len(self.times))
        self.outliers = np.zeros(len(self.times))
        
    def add_astrometric_solution(self,da,dd,mua,mud,parallax):
        
        self.wss = fcs.wss(da,dd,parallax,mua,mud,np.array(self.times),np.array(self.psi),np.array(self.pf),self.t_ref)
        
        # self.wss = (da + mua*(self.times-self.t_ref))*np.sin(self.psi) + (dd + mud*(self.times-self.t_ref))*np.cos(self.psi) + parallax*self.pf
        
    def add_jitter(self,jitter):
        self.jitter = jitter
        
    def add_planet(self, numpl, sample,Tpers=False):
        
        self.numpl = numpl
        
        
        for i in range(numpl):
            if Tpers:
                sample['Tper'+str(i)] = self.t_ref - float(sample['P'+str(i)])*float(sample['phi'+str(i)])/2/np.pi
            # print(sample[['P'+str(i),'a0'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i),'cosi'+str(i),'W'+str(i)]].values.tolist()[0])
            self.planet_orbits.append(orbit(sample[['P'+str(i),'a0'+str(i),'Tper'+str(i),'ecc'+str(i),'w'+str(i),'cosi'+str(i),'W'+str(i)]].values.tolist()[0]))
    
    def residuals(self):
        
        
        res = self.w - self.wss
        # print(res)
        for planet in self.planet_orbits:
            res -= planet.wk(self.psi,self.times)
            # print(res)
        return res
    
    def logL(self):
        
        # var = np.array(self.w_err)**2 + self.jitter**2
        logL = np.sum(np.log(norm(loc=0, scale=np.hypot(self.w_err, self.jitter)).pdf(self.residuals())))
        
        return logL

class orbit:
    
    def __init__(self,params,M0=1.,R0=1.,M1=1.,t_ref=57388.5):
        

        self.P = params[0]
        self.a0 = params[1]
        self.ecc = params[3]
        self.Tper = params[2]
        self.w =params[4]
        self.cosi = params[5]
        self.W = params[6]
        
        self.M0 = M0
        
        # self.Tper = t_ref - self.P*self.phi/2/np.pi
        
    def wk(self, psi, t):
        
        wk = fcs.wk_orb(self.P, self.Tper, self.ecc, self.cosi, self.W, self.w, self.a0, t, psi)
        
        return wk

    

def astrometry_phase_plot_logic(res, sample, sort_by_decreasing_a=False, sort_by_increasing_P=False):
    from string import ascii_lowercase
    letters = ascii_lowercase[1:]

    nplanets = int(sample[res.indices['np']])

    pj = 0
    nKOs = 0
    if res.KO:
        nKOs = res.nKO
        # ko = {letters[i]: {} for i in range(nplanets, nplanets + res.nKO)}
        params = {letters[i]: {} for i in range(nKOs)}
        nplanets += res.nKO
        for i, k in enumerate(params.keys()):
            params[k]['P'] = P = sample[res.indices['KOpars']][i]
            params[k]['φ'] = φ = sample[res.indices['KOpars']][i + 1 * res.nKO]
            params[k]['e'] = e = sample[res.indices['KOpars']][i + 2 * res.nKO]
            params[k]['a'] = a = sample[res.indices['KOpars']][i + 3 * res.nKO]
            params[k]['w'] = w = sample[res.indices['KOpars']][i + 4 * res.nKO]
            params[k]['cosi'] = cosi = sample[res.indices['KOpars']][i + 5 * res.nKO]
            params[k]['W'] = W = sample[res.indices['KOpars']][i + 6 * res.nKO]
            params[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
            params[k]['type'] = 'KO'
            params[k]['index'] = -pj - 1
            pj += 1
    else:
        params = {}
        

    planets = {letters[i]: {} for i in range(nKOs, nplanets)}
    for i, k in enumerate(planets.keys()):
        planets[k]['P'] = P = sample[res.indices['planets.P']][i]
        planets[k]['φ'] = φ = sample[res.indices['planets.φ']][i]
        planets[k]['e'] = e = sample[res.indices['planets.e']][i]
        if res.thiele_innes:
            planets[k]['A'] = A = sample[res.indices['planets.A']][i]
            planets[k]['B'] = B = sample[res.indices['planets.B']][i]
            planets[k]['F'] = F = sample[res.indices['planets.F']][i]
            planets[k]['G'] = G = sample[res.indices['planets.G']][i]
        else:
            planets[k]['a'] = a = sample[res.indices['planets.a']][i]
            planets[k]['w'] = w = sample[res.indices['planets.w']][i]
            planets[k]['cosi'] = cosi = sample[res.indices['planets.cosi']][i]
            planets[k]['W'] = W = sample[res.indices['planets.W']][i]
        planets[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
        planets[k]['type'] = 'planet'
        planets[k]['index'] = i + 1
    params.update(planets)

    keys = list(params.keys())

    if sort_by_decreasing_a:
        keys = sorted(params, key=lambda i: params[i]['a'], reverse=True)

    if sort_by_increasing_P:
        keys = sorted(params, key=lambda i: params[i]['P'])

    return nplanets, params, keys

dty = 1/365.25
dty2 = dty**2
dty3 = dty**3
twopi = 2 * np.pi

def astrometry_residuals(res, sample):
    import kima
    from kima.kepler import brandt_solver

    def ellip_rectang(t, P, e, Tper):
        M = twopi * (t - Tper) / P
        E = brandt_solver(M, e)
        X = np.cos(E) - e
        Y = np.sqrt(1-e**2) * np.sin(E)
        return X, Y

    def Thiele_Innes(a0, w, W, cosi):
        A = a0*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*cosi)
        B = a0*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*cosi)
        F = -a0*(np.sin(w)*np.cos(W) + np.cos(w)*np.sin(W)*cosi)
        G = -a0*(np.sin(w)*np.sin(W) - np.cos(w)*np.cos(W)*cosi)
        return A, B, F, G

    def ra_dec_orb(P,Tper,e,cosi,W,w,a0,t):
        A, B, F, G = Thiele_Innes(a0, w, W, cosi)
        X, Y = ellip_rectang(t, P, e, Tper)
        return B*X + G*Y, A*X + F*Y
    
    def ra_dec_orb_TI(P,Tper,e,A,B,F,G,t):
        X, Y = ellip_rectang(t, P, e, Tper)
        return B*X + G*Y, A*X + F*Y

    def wss(da,dd,par,mua,mud,t,psi,pf,tref):
        T = t - tref
        return (da + mua*dty*T)*np.sin(psi) + (dd +mud*dty*T)*np.cos(psi) +par*pf
    
    def waccels(accela,acceld,jerka,jerkd,t,psi,tref):
        T = t - tref
        return ((1/2)*accela*dty2*T**2 + (1/6)*jerka*dty3*T**3)*np.sin(psi) + ((1/2)*acceld*dty2*T**2 + (1/6)*jerkd*dty3*T**3)*np.cos(psi)

    def ra_dec_waccels(accela,acceld,jerka,jerkd,t,tref):
        T = t - tref
        return ((1/2)*accela*dty2*T**2 + (1/6)*jerka*dty3*T**3), ((1/2)*acceld*dty2*T**2 + (1/6)*jerkd*dty3*T**3)

    def wscanbias(A,theta,k,psi):
        return A*np.cos(k*(psi-theta))

    def wk_orb(P,Tper,e,cosi,W,w,a0,t,psi):
        A, B, F, G = Thiele_Innes(a0, w, W, cosi)
        X, Y = ellip_rectang(t, P, e, Tper)
        return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)
    
    def wk_orb_TI(P,Tper,e,A,B,F,G,t,psi):
        X, Y = ellip_rectang(t, P, e, Tper)
        return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)
    
    def wss_dep(da,dd,par,mua,mud,t,pfra,pfdec,tref):
        #Obtain RA and DEC values for parallax and PM plot curve
        T = t - tref
        return (da + mua*dty*T) + pfra*par,(dd +mud*dty*T)+pfdec*par
    
    def wss_dep_errs(alpha_res,dec_res,alpha_err,dec_err,da,dd,par,mua,mud,t,pf_ra,pf_dec,tref):
        #Obtain RA and DEC values and errors for parallax and PM plot points
        T = t - tref
        ra = (da + mua*dty*T) + pf_ra*par + alpha_res
        raplus = (da + mua*dty*T) + pf_ra*par + alpha_res + alpha_err
        raminus = (da + mua*dty*T) + pf_ra*par + alpha_res - alpha_err
        dec = (dd +mud*dty*T) + pf_dec*par + dec_res
        decplus = (dd +mud*dty*T) + pf_dec*par + dec_res + dec_err
        decminus = (dd +mud*dty*T) + pf_dec*par + dec_res - dec_err
        return ra,raplus,raminus,dec,decplus,decminus    

    t = np.array(res.GAIAdata.t)
    t2 = t.copy()
    wobs = np.array(res.GAIAdata.w)
    ws_err = np.array(res.GAIAdata.wsig)
    psi = np.array(res.GAIAdata.psi)
    pf = np.array(res.GAIAdata.pf)

    alpha_errs = ws_err*np.sin(psi)
    dec_errs = ws_err*np.cos(psi)

    nplanets, params, keys = astrometry_phase_plot_logic(res,sample)


    da, dd, mua, mud, par = sample[res.indices['astrometric_solution']]
    if res.n_accel_params == 4:
        accela, acceld, jerka, jerkd = sample[res.indices['accel_solution']]
    elif res.n_accel_params == 2:
        accela, acceld = sample[res.indices['accel_solution']]
        jerka, jerkd = 0, 0

    if res.scan_dep_signal:
        scan_dep_signal_params = sample[res.indices['scan_dep_signal']]
    
    wmodel = wss(da, dd, par, mua, mud, t, psi, pf, res.M0_epoch)

    #add accelerations
    if res.n_accel_params >0:
        wmodel += waccels(accela, acceld, jerka, jerkd, t, psi, res.M0_epoch)
    
    #Get full model
    for letter in keys:
        # P, phi, e, a, w, cosi, W, Tper, type, index = params[letter]
        P = params[letter]['P']
        phi = params[letter]['φ']
        e = params[letter]['e']
        if res.thiele_innes:
            A = params[letter]['A']
            B = params[letter]['B']
            F = params[letter]['F']
            G = params[letter]['G']
        else:
            a = params[letter]['a']
            w = params[letter]['w']
            cosi = params[letter]['cosi']
            W = params[letter]['W']
            A,B,F,G = Thiele_Innes(a,w,W,cosi)
        Tper = params[letter]['Tp']

        wmodel += wk_orb_TI(P, Tper, e, A, B, F, G, t, psi)
    
    #add scan-angle bias signal
    if res.scan_dep_signal:
        for j in range(res.n_scan_dep_comps):
            Ak = scan_dep_signal_params[j]
            thetak = scan_dep_signal_params[j+res.n_scan_dep_comps]
            k = 2*j + 3
            wmodel += wscanbias(Ak,thetak,k,psi)

    #get residuals
    wws = wobs - wmodel
    alpha_res, dec_res = wws * np.sin(psi), wws * np.cos(psi)

    return wws, alpha_res, dec_res

def kepmodel_analysis_periodogram(gaia_data,gaia_epoch_tcb = 2457936.875,Pmin = 0.95,Pmax = 10_000,nfreq = 50_000):
    # Initialize model
    cos_theta = np.cos(gaia_data.scan_pos_angle.values)
    sin_theta = np.sin(gaia_data.scan_pos_angle.values)

    # Create an "empty" model
    model = AstroModel(
    gaia_data.obs_time_tcb.values - gaia_epoch_tcb,
    gaia_data.centroid_pos_al.values,
    cos_theta,
    sin_theta,
    err=term.Error(gaia_data.centroid_pos_error_al.values),
    excess_noise=term.Jitter(0),
    )

    # Add 5 basic linear parameters (parallax + position + proper-motion)
    model.add_lin(gaia_data.parallax_factor_al.values, 'plx')
    model.add_lin(cos_theta, 'delta')
    model.add_lin(sin_theta, 'alpha')
    delta_time_yr = (gaia_data.obs_time_tcb.values - gaia_epoch_tcb) / 365.25
    model.add_lin(delta_time_yr * cos_theta, 'mu_delta')
    model.add_lin(delta_time_yr * sin_theta, 'mu_alpha')

    # Add scan-angle (odd) harmonics in the model
    # max_harmonics = 0: no scan-angle harmonics modeling
    # 3, 5, or 7: more and more complex modeling
    max_harmonics = 0
    for kharm in range(3, max_harmonics + 1, 2):
        model.add_lin(np.cos(kharm * gaia_data.scan_pos_angle.values), f'cos{kharm}theta')
        model.add_lin(np.sin(kharm * gaia_data.scan_pos_angle.values), f'sin{kharm}theta')

    # Add acceleration/jerk
    # max_degree_accel = 0: no-acceleration term
    # max_degree_accel = 2: acceleration
    # max_degree_accel = 3: acceleration + jerk
    max_degree_accel = 0
    for kdeg in range(2, max_degree_accel + 1):
        model.add_lin(delta_time_yr**kdeg * cos_theta, f'mu{kdeg}_delta')
        model.add_lin(delta_time_yr**kdeg * sin_theta, f'mu{kdeg}_alpha')

    # Fit all linear parameters
    model.fit_lin()

    # Adjust excess-noise (jitter)
    model.fit_param += ['cov.excess_noise.sig']
    model.fit()


    # Compute periodogram
    nu0 = 2 * np.pi / Pmax
    dnu = (2 * np.pi / Pmin - nu0) / (nfreq - 1)
    nu, power = model.periodogram(nu0, dnu, nfreq)
    P = 2 * np.pi / nu

    # Compute FAP
    kmax = np.argmax(power)
    fap = model.fap(power[kmax], nu.max())

    return P, power, kmax, fap


def kepmodel_analysis_parameters(gaia_data,gaia_epoch_tcb=2457936.875,max_harmonics=0,max_degree_accel=0,Pmin=0.95,Pmax=10000,nfreq=50000,fap_threshold=0.001):
    # Initialize model
    cos_theta = np.cos(gaia_data.scan_pos_angle.values)
    sin_theta = np.sin(gaia_data.scan_pos_angle.values)

    # Create an "empty" model
    model = AstroModel(
    gaia_data.obs_time_tcb.values - gaia_epoch_tcb,
    gaia_data.centroid_pos_al.values,
    cos_theta,
    sin_theta,
    err=term.Error(gaia_data.centroid_pos_error_al.values),
    excess_noise=term.Jitter(0),
    )

    # Add 5 basic linear parameters (parallax + position + proper-motion)
    model.add_lin(gaia_data.parallax_factor_al.values, 'plx')
    model.add_lin(cos_theta, 'delta')
    model.add_lin(sin_theta, 'alpha')
    delta_time_yr = (gaia_data.obs_time_tcb.values - gaia_epoch_tcb) / 365.25
    model.add_lin(delta_time_yr * cos_theta, 'mu_delta')
    model.add_lin(delta_time_yr * sin_theta, 'mu_alpha')

    # Add scan-angle (odd) harmonics in the model
    # max_harmonics = 0: no scan-angle harmonics modeling
    # 3, 5, or 7: more and more complex modeling
    for kharm in range(3, max_harmonics + 1, 2):
        model.add_lin(np.cos(kharm * gaia_data.scan_pos_angle.values), f'cos{kharm}theta')
        model.add_lin(np.sin(kharm * gaia_data.scan_pos_angle.values), f'sin{kharm}theta')

    # Add acceleration/jerk
    # max_degree_accel = 0: no-acceleration term
    # max_degree_accel = 2: acceleration
    # max_degree_accel = 3: acceleration + jerk
    for kdeg in range(2, max_degree_accel + 1):
        model.add_lin(delta_time_yr**kdeg * cos_theta, f'mu{kdeg}_delta')
        model.add_lin(delta_time_yr**kdeg * sin_theta, f'mu{kdeg}_alpha')

    # Fit all linear parameters
    model.fit_lin()

    # Adjust excess-noise (jitter)
    model.fit_param += ['cov.excess_noise.sig']
    model.fit()
    logL_baseline = model.loglike()
    BIC_baseline = -2 * model.loglike() + len(model.fit_param) * np.log(model.n)

    metrics = [logL_baseline,BIC_baseline]
    # Compute periodogram
    nu0 = 2 * np.pi / Pmax
    dnu = (2 * np.pi / Pmin - nu0) / (nfreq - 1)
    nu, power = model.periodogram(nu0, dnu, nfreq)
    P_prdgm = 2 * np.pi / nu

    # Compute FAP
    kmax = np.argmax(power)
    fap = model.fap(power[kmax], nu.max())

    # Add the planet in the model if fap better than threshold
    planet = False
    if fap<fap_threshold:
        model.add_keplerian_from_period(P_prdgm[kmax], name='b')
        model.set_keplerian_param('b', ['P', 'as', 'M0', 'e', 'w', 'i', 'bigomega'])
        model.fit()
        logL_keplerian = model.loglike()
        BIC_keplerian = -2 * model.loglike() + len(model.fit_param) * np.log(model.n)
        metrics += [logL_keplerian,BIC_keplerian]
        planet = True

    fit_params, fit_params_error = model.get_param_error()

    plx,dd,da,mud,mua = fit_params[:5]
    plx_error,dd_error,da_error,mud_error,mua_error = fit_params_error[:5]

    baseline_params = [da,dd,mua,mud,plx]
    baseline_errors = [da_error,dd_error,mua_error,mud_error,plx_error]

    add_index = 5+1
    if max_harmonics != 0:
        add_index += int((max_harmonics-1)/2)*2
    if max_degree_accel != 0:
        add_index += (max_degree_accel-1)*2

    if planet:
        P,a,phi,e,w,i,W = fit_params[add_index:add_index+7]
        P_errors,a_errors,phi_errors,e_errors,w_errors,i_errors,W_errors = fit_params_error[add_index:add_index+7]
        planet_params = [P,phi,e,a,w,i,W]
        planet_errors = [P_errors,phi_errors,e_errors,a_errors,w_errors,i_errors,W_errors]
    else:
        planet_params = []
        planet_errors = []

    return_params = [P_prdgm, power, kmax, fap, metrics, baseline_params, baseline_errors, planet, planet_params, planet_errors]

    return return_params


