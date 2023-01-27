#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:11:27 2022

@author: thomasbaycroft
"""

import numpy as np 
import scipy as sp
from astropy import constants
pi = np.pi
Ms = constants.M_sun.value
c = constants.c.value
G = constants.G.value
Rs = constants.R_sun.value
Mj = constants.M_jup.value
AU = constants.au.value
Me = constants.M_earth.value

#################################################

def keplerian(time, p, k, ecc, omega, t0, vsys):
    '''
    Obtains the radial velocity due to a keplerain orbit for an array of times time, given 
    Period P, semi-amplitude k, eccentricity ecc, argument of pericentre omega, time of 
    preicentre passage t0 and systemic velocity vsys.
    Returns the array (same size as time) of radial velocities
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, t0 = np.atleast_1d(p, k, ecc, omega, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            vel += k[i] * (np.cos(omega[i]+nu)+ ecc[i]*np.cos(omega[i]))
        vel += vsys
    return vel

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

def keplerian_plus(time, p, k, ecc, omega, omdot, t0, vsys):
    '''
    Extension of the keplerian function allowing for linearapsidal precession of the orbit
    requires an extra input, the rate of apsidal precession omdot
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, omdot, t0 = np.atleast_1d(p, k, ecc, omega, omdot, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            p[i] = Period_change(p[i], ecc[i], omdot[i])
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            wdot = omdot[i]*(4.8481*(0.000001)/365.25) #convert from arcsec per year to radians per day
            w = omega[i]+wdot*(time-t0[i])
            vel += k[i] * (np.cos(nu+w) + ecc[i]*np.cos(w))
        vel += vsys 
    return vel

def keplerian_plusplus(time, p, k, ecc, omega, omdot, t0, vsys, M0, R0, M1, relativistic_correction, tidal_correction):
    '''
    Extension to keplerian_plus which includes relativistic and tidal correction, requires
    additional inputs of the Massas of the two bodies, M0 and M1 the radius of the observed 
    body (needed for the tidal correction) and boolians for where to do the relativisdtic and/or
    the tidal corrections.
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, omdot, t0 = np.atleast_1d(p, k, ecc, omega, omdot, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            p[i] = Period_change(p[i], ecc[i], omdot[i])
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            wdot = omdot[i]*(4.8481*(0.000001)/365.25) #convert from arcsec per year to radians per day
            w = omega[i]+wdot*(time-t0[i])
            vel += k[i] * (np.cos(nu+w) + ecc[i]*np.cos(w))
            extra_vel = post_newtonian(k[i], nu, ecc[i], w, p[i], M0, R0, M1, relativistic_correction, tidal_correction)

        vel += vsys + extra_vel
    return vel

def post_newtonian(K0, nu, ecc, w, P, M0, R0, M1, relativistic_correction, tidal_correction):
    '''
    Calculates the relativistic and tidal corrections to the radial velocity
    '''
    sini = 1
    v = 0.0
    if relativistic_correction or tidal_correction:
        if M1 == None:
            K1 = get_K2(K0,M0,P,ecc)
            M1 = M0*K0/K1
        else:
            K1 = K0*M0/M1
    if tidal_correction:
        if R0 == None:
            R0 = M0**0.8
        delta_v_tide = v_tide(R0, M0, P, nu, w, M1, "M_s")
        v += delta_v_tide
    if relativistic_correction:
        cl = c
        delta_LT = K0**2*np.sin(nu+w)**2*(1+ecc*np.cos(nu))/cl
        delta_TD = K0**2*(1 + ecc*np.cos(nu) - (1-ecc**2)/2)/(cl*sini**2)
        delta_GR = K0*(K0+K1)*(1+ecc*np.cos(nu))/(cl*sini**2)
        v += delta_LT + delta_TD + delta_GR
    
    return v

def Period_change(P2,e,wdot):
    '''
    Gives the actual period since wdot overestimates by a slight factor
    We take off from the period, the time taken due to the extra apsidal motion

    Assume spread evenly/circular orbit
    '''
    return P2*(1+wdot/(3600*180*365.25*2)*P2)

def v_tide(R_star, M_star, P, f, w, M_p, M_p_unit):
    '''
    https://arxiv.org/pdf/1107.6005.pdf Equation 17

    Takes:
        R_star = Radius of star (solar radius) float
        M_star = Mass of star (solar masses) float
        P = Period (days) float
        f = Mean motion
        phi_0 = 
        M_p = Mass of planet (or orbiting body) (M_j) float
        M_p_unit = Unit "M_j"/"M_s" str

    returns:
        v_tide = Velocity of tide (m/s) float
        
    '''
    
    phi_0 = np.pi/2 - w #phi_0 defined relative to argument of pericenter
    
    M_ratio = Ms/Mj
    if M_p_unit == "M_s":
        M_p *= M_ratio
        
    v_tide = ((((1.13 * M_p)/(M_star * (M_star + (M_p/M_ratio)))) * (R_star**4)) / P**3)*np.sin(2*(f-phi_0))

    return v_tide

def masses(SB2,K,P,e,M1=1,q=1):
    '''
    Returns the masses of the primary and secondary star, given:
        K: Semi-amplitude (m/s)
        P: Period (days)
        e: eccentricity
        M1: Primary mass (M_sun) used if not SB2
        q: mass ratio (used if SB2)
        SB2: boolean for whether a double lined binary or no (decides which method)
    If SB2 you get the dynamical masses for both components, if not then only the secondary 
    mass is calculated, relative to the primary
    '''    
    if SB2:
        mass_func = (P*24*3600)*K**3*(1-e**2)**(3/2)/2/np.pi/G
        M1 = (mass_func*(1+q)**2)/Ms
        M2 = M1*q
    else:
        M2 = get_mass(K,M1,P,e)
            
    return M1,M2

def semi_amplitude(M0,M1,P,e):
    '''
    
    from Lovis and Fischer 
    Parameters
    ----------
    M0 : float
        Body 0 mass (Ms)
    M1 : float
        Body 1 mass (Ms)
    P : float
        Orbital Period (days)
    e : float
        eccentricity

    Returns
    -------
    float
        semiamplitude of RV signal on body 0 due to body 1 (m/s)

    '''
    m1 = M1*1047.5655
    m01 = M0 + M1
    return 28.4329*(1-e**2)**(-1/2)*(m1)*(m01)**(-2/3)*(P/365.25)**(-1/3)

def get_mass(K,M0,P,e):
    '''
    Root finder for the Mass of a body, given:
        M0: Central mass (M_sun)
        K: semi-amplitude (m/s)
        P: Period (days)
        e: eccentricity
    '''
    M = []
    K,P,e = np.atleast_1d(K,P,e)
    #if M0 is not an array make it so, with the same dimensions as the other parameters
    if type(M0) in [float,int]:
        M0 = np.ones(len(K))*M0
    else:
        M0 = np.atleast_1d(M0)
    # print(K)
    for i in range(K.size):
        a = 0
        b = M0[i]
        c = M0[i]/2
        eps = semi_amplitude(M0[i],b,P[i],e[i]) - semi_amplitude(M0[i],a,P[i],e[i])
        while eps>0.01 or eps < -0.01: 
            c = (b+a)/2
            x = K[i] - semi_amplitude(M0[i],c,P[i],e[i])
            if x < 0:
                b = c
            else:
                a = c
            eps = semi_amplitude(M0[i],b,P[i],e[i]) - semi_amplitude(M0[i],a,P[i],e[i])
        M.append(c)
    return M
# def get_mass(K,M,P,e):
    
#     M2 = sp.optimize.root(semi_amplitude())
def get_K2(K1, M, P, ecc):
    '''
    Newton-raphson root finder for get the semi-amplitude that one would get from Body 2 give:
        K1: semi-amplitude of Body 1 (m/s)
        M: Mass of Body 1 (M_sun?)
        P: period of orbit (days)
        ecc: eccentricity of orbit
    '''
    #Newton-raphson
    M_est = mass(K1,P,ecc,M*Ms)
    k = semi_amplitude(M,M_est,P,ecc)
    while abs(k-K1)>50:
        M_est -= f_M(K1,M,M_est,P,ecc)/f_dash_M(K1,M,M_est,P,ecc)
        k = semi_amplitude(M,M_est,P,ecc)

    K2 = K1*M/M_est
    
    return K2

def mass(K,P,e=0,M0=Ms):
    '''
    from Lovis & Fischer, mass estimate for secondary ignoring its own mass
    '''

    return ((K/28.4329)*(1-e**2)**(1/2)*(M0/Ms)**(2/3)*(P/365)**(1/3))*Mj/Ms

def f_M(K1, M0, M1, P, ecc):
    '''
    function for newton-raphson
    '''
    return semi_amplitude(M0, M1, P, ecc) - K1
    
def f_dash_M(K, M0, M1, P, ecc):
    '''
    function derivativbe for Newton-raphson
    M0 and M1 in solar mass, P in days returns in m/s
    '''
    MjMs = 1047.5655
    m1 = M1*MjMs
    m01 = M0 + M1
    
    f_dash_1 = 28.4329*(1-ecc**2)**(-1/2)*MjMs*m01**(-2/3)*(P/365)**(-1/3)
    f_dash_2 = -2/3*28.4329*(1-ecc**2)**(-1/2)*m1*m01**(-5/3)*(P/365)**(-1/3)
    
    return f_dash_1 - f_dash_2

def sma_tot(P,M0,M1=0):
    '''
    Calculates the semi-major axis of an orbit given:
        P: Period (days)
        M0: Central mass (M_sun)
        M1: Mass or second body (M_sun)
    This uses Kepler's 3rd law, so sem-major axis is that of the combined 
    orbit rather than of an individual body around the centre of mass
    '''
    return ((P*24*3600)**2*G*(M0+M1)*Ms/(4*np.pi**2))**(1/3)/AU

def sma_2body(P,M0,M1):
    '''
    Calculates the semi-major axes of an orbit given:
        P: Period (days)
        M0: Central mass (M_sun)
        M1: Mass or second body (M_sun)
    Returns the semi-major axis of each body's orbit about the centre of mass
    '''
    a_tot = sma_tot(P,M0,M1)
    a0 = a_tot*M1/(M0+M1)
    a1 = a_tot-a0
    return a0,a1

def P_from_sma(a,M0,M1=0):
    '''
    Calculates the speriod of an orbit given:
        a: sma (AU)
        M0: Central mass (M_sun)
        M1: Mass or second body (M_sun)
    This uses Kepler's 3rd law, so sem-major axis should be that of the combined 
    orbit rather than of an individual body around the centre of mass
    '''
    return ((a*AU)**3*4*pi**2/G/(M0+M1)/Ms)**(1/2)/24/3600

#################################################
#Calculation of wdot, the apsidal precession rate, due the dynamics of a 3rd body,
#the rotational effect and due to GR (in the assumtion of an eclipsing binary and a
#transiting planet) as in Baycroft et al. 2023
#################################################

def wdot_dyn(M0,M1,M2,e1,e2,P1,P2,w1,inc):
    '''
    Calculate the rate of apsidal precession of an orbit due to 
    an external perturber given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        M2: Mass of perturber (M_sun)
        e1: eccentricity of outer orbit
        e2: eccentricity of inner orbit
        P1: orbital period of outer orbit (days)
        P2: orbital period of inner orbit (days)
        w1: argument of pericentre of inner orbit (radians)
        inc: mutual inclination between orbits, i.e. inclination to line 
        of sight of outer orbit for an eclipsing binary (radians) 
    This is under the approximation of an eclipsing and transiting 
    system (inclination is small)
    Returns rate of precession in arcsec/yr
    '''
    
    a1 = sma_tot(P1,M0,M1)*AU
    a2 = sma_tot(P2,M0+M1,M2)*AU
    
    m0 = M0*Ms
    m1 = M1*Ms
    m2 = M2*Ms
    
    C2 = G/16*m0*m1*m2*a1**2/(m0+m1)/((1-e2**2)**(3/2))/(a2**3)
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = 12*C2/G1*(1-e1**2)*(1 + (1-5*np.cos(w1)*np.cos(w1))*np.sin(inc)*np.sin(inc))
    wdot *= 180/np.pi*3600*3600*24*365.25
    
    return wdot

def wdot_rot(M0,M1,e1,P1,k0,k1,Omega0,Omega1,R0,R1):
    '''
    Calculate the rate of apsidal precession of an orbit due to 
    rotational effects given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e1: eccentricity of outer orbit
        P1: orbital period of outer orbit (days)
        k0: second Love number for the primary body
        k1: second Love number for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
    Returns rate of precession in arcsec/yr
    '''
    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Ms
    m1 = M1*Ms
    R0 *= Rs
    R1 *= Rs
    
    J20 = k1*Omega0**2*R0**3/3/G/m0
    J21 = k1*Omega1**2*R1**3/3/G/m1
    
    C0 = G*m0*m1*J20*R0**2/2/a1**3
    C1 = G*m0*m1*J21*R1**2/2/a1**3
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = 3*(C0+C1)/G1/((1-e1**2)**(3/2))
    wdot *= 180/np.pi*3600*3600*248365.25
    
    return wdot

def wdot_GR(M0,M1,e1,P1):
    '''
    Calculate the rate of apsidal precession of an orbit due to GR given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e1: eccentricity of outer orbit
        P1: orbital period of outer orbit (days)
    Returns rate of precession in arcsec/yr
    '''
    
    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Ms
    m1 = M1*Ms
    
    Cg = 3*G**2*m0*m1*(m0+m1)/(a1**2*c**2)
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = Cg/G1/((1-e1**2)**(1/2))
    wdot *= 3600*24*365.25*180*3600/np.pi
    
    return wdot

def planet_mass(M0,M1,e1,e2,P1,P2,w1,inc,wdot):
    

    wdot -= wdot_GR(M0,M1,e1,P1)
    
    m0 = 0.000000001
    m1 = 10
    wd0 = wdot_dyn(M0,M1,m0,e1,e2,P1,P2,w1,inc)
    wd1 = wdot_dyn(M0,M1,m1,e1,e2,P1,P2,w1,inc)
    Dwd = wd1 - wd0
    i = 0
    while abs(Dwd) >= 0.1:
        m2 = (m1+m0)/2
        wd2 = wdot_dyn(M0,M1,m2,e1,e2,P1,P2,w1,inc)
        if (wd2 - wdot)*(wd0-wdot) < 0:
            m1 = m2
        else:
            m0 = m2
        wd0 = wdot_dyn(M0,M1,m0,e1,e2,P1,P2,w1,inc)
        wd1 = wdot_dyn(M0,M1,m1,e1,e2,P1,P2,w1,inc)
        Dwd = wd1 - wd0

    return m2

#################################################
#other functions
#################################################


def T_stab(M0,M1,a1,a2,e2,P2):
    '''
    Calculate stability timescale analystically following Zhang, Naoz and Will 2023
    Assuems binary has circular orbit, and massless planet
    Units of M0 and M1 should be the same, as with a1 and a2
    returns stability timescale in the units of P2
    '''
    
    q = M1/M0
    alpha = (1-e2)*a2/a1
    
    return P2*q**(-2)/4/(1+e2)*alpha**7*(1-e2**2)*(1+q)**4

def get_Tstab(M0,M1,P1,P2,e2):
    
    a1 = sma_2body(P1, M0, M1)[1]
    a2 = sma_tot(P2, M0+M1)
    
    return T_stab(M0,M1,a1,a2,e2,P2)

def H_W_stab_lim(a,e):
    return 2.278 + 3.824*e - 1.71*e**2

def H_W_stab_lim_ext(a,e,mu):
    return 1.6 +5.1*e - 2.22*e**2 + 4.12*mu - 4.27*e*mu - 5.09*mu**2

def get_HW_stab(M0,M1,P,e,ext=True):
    
    a = sma_tot(P,M0,M1)
    mu = min(M1/(M0+M1),M0/(M0+M1))
    if ext:
        a_pl = H_W_stab_lim_ext(a, e, mu)*a
    else:
        a_pl = H_W_stab_lim(a, e)*a
    
    P_pl = P_from_sma(a_pl, M0+M1)
    
    return P_pl

# import matplotlib.pyplot as plt
# import time
# wdotmu = 0.186*3600
# wdotsig = 0.056*3600
# number = 10000
# P = np.random.rand(number)*800 + 40
# wdot = np.random.randn(number)*wdotsig + wdotmu
# t0 = time.time()
# for i in range(number):
#     plt.scatter(P[i],planet_mass(2.151,1.992,0.2419,0,8.289761,P[i],4.36,0,wdot[i])*Ms/Mj,c='black',s=8,alpha=0.1)
# plt.xlabel('Period (days)')
# plt.ylabel(r'Mass (M$_J$)')
# # plt.ylim([np.log10(0.1),np.log10(20)])
# plt.semilogy()
# xax = np.linspace(40,840,60)
# Mbin = np.ones(len(xax))*(2.151+1.992)
# es = np.ones(len(xax))*0.2419
# Ks = np.ones(len(xax))*5000
# ms = np.array(get_mass(Ks,Mbin,xax,es))*Ms/Mj
# plt.plot(xax,ms,c='b',label='5 km/s')
# Ks = np.ones(len(xax))*1000
# ms = np.array(get_mass(Ks,Mbin,xax,es))*Ms/Mj
# plt.plot(xax,ms,c='r',label='1 km/s')
# Ks = np.ones(len(xax))*400
# ms = np.array(get_mass(Ks,Mbin,xax,es))*Ms/Mj
# plt.plot(xax,ms,c='g',label='400 m/s')
# plt.axhline(80,ls='--',c='grey')
# plt.axhline(13,ls='--',c='grey')
# plt.legend()
# print(time.time() - t0)