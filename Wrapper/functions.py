#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:11:27 2022

@author: thomasbaycroft
"""

import numpy as np 
import scipy as sp
from astropy import constants
from PyAstronomy.pyTiming import pyPeriod

pi = np.pi
c = constants.c.value
G = constants.G.value
AU = constants.au.value
Rsun = constants.R_sun.value
Msun = constants.M_sun.value
Mj = constants.M_jup.value
Me = constants.M_earth.value
Re = constants.R_earth.value
Mnep = 1.0241*10**(26)
Msat = 5.68319*10**(26)

#################################################
#Radial Velocity functions
#################################################

def keplerian(time, p, k, ecc, omega, t0, vsys):
    '''
    keplerian function given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
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

def keplerian_phase(phase, p, k, ecc, omega, t0, vsys):
    '''
    keplerian function given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
    Returns the array (same size as time) of radial velocities
    '''
    vel = np.zeros_like(phase)
    p, k, ecc, omega, t0 = np.atleast_1d(p, k, ecc, omega, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            # M = 2.*pi * (time-t0[i]) / p[i]
            M = phase*2*np.pi
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            vel += k[i] * (np.cos(omega[i]+nu)+ ecc[i]*np.cos(omega[i]))
        vel += vsys
    return vel

def true_anomaly(E, e):
    '''
    calculate true anomaly from the eccentric anomaly
    '''
    return 2. * np.arctan( np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.))

def ecc_anomaly(M, e):
    '''
    calculate the eccentric anomaly from the mean anomaly
    '''
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

def mean_anom(f,e):
    '''
    calculate mean anomaly from the true anomaly
    '''
    E = 2. * np.arctan( np.sqrt((1.-e)/(1.+e)) * np.tan(f/2.))

    M = E - e*np.sin(E)
    return M

def ellip_rectang(t,P,e,Tper):
    
    M = 2.*pi * (t-Tper)/P
    E = ecc_anomaly(M, e)
    
    X = np.cos(E) - e
    Y = (1-e**2)**(0.5)*np.sin(E)
    
    return X,Y


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
    keplerian function with a linear apsidal precession rate term added, given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        omdot: rate of change of omega (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
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
    RV keplerian function with a linear apsidal precession term added and the option to include relativistic and tidal corrections, given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        omdot: rate of change of omega (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
        M0, R0: Mass and Radius of primary body (on which RV data is obtained) (floats)
        M1: Mass of secondary body (float)
        relativistic_correction: flag to turn on relativistic correction (bool)
        tidal_correction: flag to turn on Tidal correction (bool)
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

def post_newtonian(K0, nu, ecc, w, P, M0, R0, M1, relativistic_correction, tidal_correction,sini=1):
    '''
    Calculates the post-newtonian contributions as in Baycroft et. al 2023
    '''
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
    Gives the anomalistic period since wdot overestimates by a slight factor
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
    
    M_ratio = Msun/Mj
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
        mass_func = (P*24*3600)*(K/q)**3*(1-e**2)**(3/2)/2/np.pi/G
        M1 = (mass_func*(1+q)**2)/Msun
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
        Body 0 mass (Msun)
    M1 : float
        Body 1 mass (Msun)
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

def semi_amplitude_2(M0,M1,P,e,sini=1):
    '''
    Parameters
    ----------
    M0 : float
        Body 0 mass (Msun)
    M1 : float
        Body 1 mass (Msun)
    P : float
        Orbital Period (days)
    e : float
        eccentricity

    Returns
    -------
    float
        semiamplitude of RV signal on body 0 due to body 1 (m/s)

    '''
    
    return (2*np.pi*G/(P*24*3600))**(1/3)*M1*Msun*sini*((M0+M1)*Msun)**(-2/3)*(1-e**2)**(-0.5)

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
    # print(M0)
    # print(P)
    # print(e)
    for i in range(K.size):
        a = 0
        b = 2*M0[i]
        c = b/2
        eps = semi_amplitude(M0[i],b,P[i],e[i]) - semi_amplitude(M0[i],a,P[i],e[i])
        while eps>0.000000001 or eps < -0.0000000001: 
            c = (b+a)/2
            x = K[i] - semi_amplitude(M0[i],c,P[i],e[i])
            if x < 0:
                b = c
            else:
                a = c
            eps = semi_amplitude(M0[i],b,P[i],e[i]) - semi_amplitude(M0[i],a,P[i],e[i])
        M.append(c)
    return M

def get_M1_Yaz_M2(M2,P,K1,e,i=np.pi/2):
    '''
    Derive the value of M1, from a measurement of M2 as in Davis et al. 2023, given:
        M2: secondary star mass (M_sun)
        P: orbital period (days)
        K1: RV semi-amplitude of the primary star orbit (m/s)
        e: eccentricity of the binary orbit
        i: inclination to the line of sight (assumed to be 90 if not specified) (rad)
    Returns mass of primary star (M_sun)
    '''
    m2 = M2*Msun
    C = (1-e**2)**(3/2)/(np.sin(i)**3)*(P*24*3600)*(K1**3)/2/np.pi/G
    return (m2**(3/2)*C**(-1/2) - m2)/Msun

def get_mass_scp(K,M0,P,e):
    '''
    Root finder for the Mass of a body, given:
        M0: Central mass (M_sun)
        K: semi-amplitude (m/s)
        P: Period (days)
        e: eccentricity
    Uses scipy.optimize.root
    '''
    
    def fun(M2):
        return semi_amplitude(M0,M2,P,e) - K    
    Mass = sp.optimize.root(fun,M0)
    if Mass.message != 'The solution converged.':
        print('Warning: Mass not conerged with message',Mass.message)
    return Mass.x[0]

    
def get_K2(K1, M, P, ecc):
    '''
    Newton-raphson root finder for get the semi-amplitude that one would get from Body 2 give:
        K1: semi-amplitude of Body 1 (m/s)
        M: Mass of Body 1 (M_sun?)
        P: period of orbit (days)
        ecc: eccentricity of orbit
    '''
    #Newton-raphson
    M_est = mass(K1,P,ecc,M*Msun)
    k = semi_amplitude(M,M_est,P,ecc)
    while abs(k-K1)>50:
        M_est -= f_M(K1,M,M_est,P,ecc)/f_dash_M(K1,M,M_est,P,ecc)
        k = semi_amplitude(M,M_est,P,ecc)

    K2 = K1*M/M_est
    
    return K2

def mass(K,P,e=0,M0=Msun):
    '''
    from Lovis & Fischer, mass estimate for secondary ignoring its own mass
    '''

    return ((K/28.4329)*(1-e**2)**(1/2)*(M0/Msun)**(2/3)*(P/365)**(1/3))*Mj/Msun

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

#################################################
#Orbital element functions
#################################################

def sma_tot(P,M0,M1=0):
    '''
    Calculates the semi-major axis of an orbit given:
        P: Period (days)
        M0: Central mass (M_sun)
        M1: Mass or second body (M_sun)
    This uses Kepler's 3rd law, so sem-major axis is that of the combined 
    orbit rather than of an individual body around the centre of mass
    returns a: semimajor axis (AU)
    '''
    return ((P*24*3600)**2*G*(M0+M1)*Msun/(4*np.pi**2))**(1/3)/AU

def sma_2body(P,M0,M1):
    '''
    Calculates the semi-major axes of an orbit given:
        P: Period (days)
        M0: Central mass (M_sun)
        M1: Mass or second body (M_sun)
    Returns the semi-major axis of each body's orbit about the centre of mass (AU)
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
    return ((a*AU)**3*4*pi**2/G/(M0+M1)/Msun)**(1/2)/24/3600

def frame_trans(Omega,I,w):
    '''
    Transformation fromorbital frame x,y,z to reference frame X,Y,Z
        Omega: longitude of ascending node (rad)
        I: inclination (rad)
        w: argument of pericentre (rad)
    returns
        T: transfomration matrix
    '''

    R1 = np.array([[np.cos(Omega),-np.sin(Omega),0],[np.sin(Omega),np.cos(Omega),0],[0,0,1]])
    R2 = np.array([[1,0,0],[0,np.cos(I),-np.sin(I)],[0,np.sin(I),np.cos(I)]])
    R3 = np.array([[np.cos(w),-np.sin(w),0],[np.sin(w),np.cos(w),0],[0,0,1]])

    T1 = np.matmul(R2,R3)
    T = np.matmul(R1,T1)

    return T

def frame_trans_inv(Omega,I,w):
    '''
    Transformation from reference frame X,Y,Z to orbital frame x,y,z
        Omega: longitude of ascending node (rad)
        I: inclination (rad)
        w: argument of pericentre (rad)
    returns
        Tinv: transfomration matrix
    '''

    T = frame_trans(Omega,I,w)
    Tinv = np.linalg.inv(T)

    return Tinv

def I_xyz(x,y,z,dx,dy,dz):
    '''
    Orbital inclination calculated from the positions and velocities
    '''
    h = ((y*dz - z*dy)**2 + (z*dx - x*dz)**2 + (x*dy - y*dx)**2)**(0.5)
    hz = x*dy - y*dx
    I = np.arccos(hz/h)
    return I

def W_xyz(x,y,z,dx,dy,dz):
    '''
    logitude of ascending node calculated from the positions and velocities
    '''
    hx = y*dz - z*dy
    hy = z*dx - x*dz
    arg = -hx/hy
    Omega = np.arctan2(-hx,hy)
    return Omega


#################################################
#All things wdot:
#Calculating the apsidal precession rate due the dynamics of a 3rd body,
#the rotational effect and due to GR (in the assumtion of an eclipsing binary and a
#transiting planet) as in Baycroft et al. 2023
# ETVs from apsidal precession
#################################################

asyr_to_radday = np.pi/180/3600/365.25

def wdot_dyn_full(M0,M1,M2,e1,e2,P1,P2,w1,W1,W2,I1,I2):
    '''
    Calculate the rate of apsidal precession of an orbit due to 
    an external perturber given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        M2: Mass of perturber (M_sun)
        e1: eccentricity of inner orbit
        e2: eccentricity of outer orbit
        P1: orbital period of inner orbit (days)
        P2: orbital period of outer orbit (days)
        w1: argument of pericentre of inner orbit (radians)
        W1: Longitude of ascending node of inner orbit (rad)
        W2: Longitude of ascending node of outer orbit (rad)
        I1: inclination of inner orbit (rad)
        I2: inclination of outer orbit (rad)
    Returns rate of precession in arcsec/yr
    '''
    a1 = sma_tot(P1,M0,M1)*AU
    a2 = sma_tot(P2,M0+M1,M2)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    m2 = M2*Msun
    
    C2 = G/16*m0*m1*m2*a1**2/(m0+m1)/((1-e2**2)**(3/2))/(a2**3)
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    cosi = np.sin(I1)*np.sin(I2)*np.cos(W1-W2) + np.cos(I1)*np.cos(I2)
    
    cosa = np.sin(I1)*np.sin(w1)*np.cos(I2) - np.sin(I2)*np.cos(w1)*np.sin(W1-W2) - np.sin(I2)*np.cos(I1)*np.sin(w1)*np.cos(W1-W2)
    
    dcidI = np.cos(I1)*np.sin(I2)*np.cos(W1-W2) - np.sin(I1)*np.cos(I2)
    dcadI = np.cos(I1)*np.sin(w1)*np.cos(I2) + np.sin(I2)*np.sin(I1)*np.sin(w1)*np.cos(W1-W2)
    
    dhde = C2*(-24*e1 + 12*e1*cosi**2 + 60*e1*cosa**2)
    dhdi = C2*(-12*(1-e1**2)*cosi*dcidI + 60*e1**2*cosa*dcadI)
    
    wdot = -(1-e1**2)/e1/G1*dhde + np.cos(I1)/np.sin(I1)/G1*dhdi
    
    wdot *= 180/np.pi*3600*3600*24*365.25
    
    return wdot
    

def wdot_dyn(M0,M1,M2,e1,e2,P1,P2,w1,inc):
    '''
    Calculate the rate of apsidal precession of an orbit due to 
    an external perturber given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        M2: Mass of perturber (M_sun)
        e1: eccentricity of inner orbit
        e2: eccentricity of outer orbit
        P1: orbital period of inner orbit (days)
        P2: orbital period of outer orbit (days)
        w1: argument of pericentre of inner orbit (radians)
        inc: mutual inclination between orbits, i.e. inclination to line 
        of sight of outer orbit for an eclipsing binary (radians) 
    This is under the approximation of an eclipsing and transiting 
    system (inclination is small)
    Returns rate of precession in arcsec/yr
    '''
    
    a1 = sma_tot(P1,M0,M1)*AU
    a2 = sma_tot(P2,M0+M1,M2)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    m2 = M2*Msun
    
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
        e1: eccentricity of inner orbit
        P1: orbital period of inner orbit (days)
        k0: second Love number for the primary body
        k1: second Love number for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
    Returns rate of precession in arcsec/yr
    '''
    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    R0 *= Rsun
    R1 *= Rsun
    
    J20 = k0*Omega0**2*R0**3/3/G/m0
    J21 = k1*Omega1**2*R1**3/3/G/m1
    
    C0 = G*m0*m1*J20*R0**2/2/a1**3
    C1 = G*m0*m1*J21*R1**2/2/a1**3
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = 3*(C0+C1)/G1/((1-e1**2)**(3/2))
    wdot *= 180/np.pi*3600*3600*24*365.25
    
    return wdot

def wdot_rot_full(M0,M1,e1,P1,k0,k1,Omega0,Omega1,R0,R1,I,Omega,eps0,eps1,phi0,phi1):
    '''
    Calculate the rate of apsidal precession of an orbit due to 
    rotational effects inclusing misaligned spins given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e1: eccentricity of inner orbit
        P1: orbital period of inner orbit (days)
        k0: second Love number for the primary body
        k1: second Love number for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
        I: orbital inclination  (rad) (pi/2 if eclipsing)
        Omega: longitude of ascending node (rad)
        eps0: angle between the equator of primary and the plane of the sky (rad)
        eps1: angle between the equator of secondary and the plane of the sky (rad)
        phi0: angle between the the x reference axis in the plane of the sky and the line of nodes between this plane and the equator of primary (spin precession angle) (rad)
        phi1: spin precession angle of secondary (rad)

    Returns rate of precession in arcsec/yr
    '''

    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    R0 *= Rsun
    R1 *= Rsun
    
    J20 = k0*Omega0**2*R0**3/3/G/m0
    J21 = k1*Omega1**2*R1**3/3/G/m1

    ct0 = np.sin(I)*np.sin(eps0)*np.cos(Omega - phi0) + np.cos(I)*np.cos(eps0)
    ct1 = np.sin(I)*np.sin(eps1)*np.cos(Omega - phi1) + np.cos(I)*np.cos(eps1)

    st0 = (1 - ct0**2)**(0.5)
    st1 = (1 - ct1**2)**(0.5)

    P2_0 = (3*ct0**2 - 1)/2
    P2_1 = (3*ct1**2 - 1)/2

    dP2_0 = 3*ct0*(np.cos(I)*np.sin(eps0)*np.cos(Omega - phi0) - np.sin(I)*np.cos(eps0))
    dP2_1 = 3*ct1*(np.cos(I)*np.sin(eps1)*np.cos(Omega - phi1) - np.sin(I)*np.cos(eps1))

    C0 = G*m0*m1*J20*R0**2*P2_0/2/(a1**3)
    C1 = G*m0*m1*J21*R1**2*P2_1/2/(a1**3)
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = 3*(C0+C1)/G1/((1-e1**2)**(3/2)) - np.cos(I)/np.sin(I)/G1/((1-e1**2)**(3/2))*(C0/P2_0*dP2_0 + C1/P2_1*dP2_1)
    wdot *= 180/np.pi*3600*3600*24*365.25
    
    return wdot


def wdot_spin_maxneg(M0,M1,e,P,R0,R1,a,Omega0,Omega1,k0,k1,inc):
    '''
    Calculate the maximum retrograde rate of apsidal precession of an orbit due to 
    misaligned spins given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e: eccentricity of inner orbit
        P: orbital period of inner orbit (days)
        k0: apsidal constant for the primary body
        k1: apsidal constant for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        inc: inclination of the orbit relative to the observer
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
        a: semi-major axis of orbit (AU)
    Returns rate of precession in arcsec/yr
    '''
    r0 = R0*Rsun/(a*AU)
    r1 = R1*Rsun/(a*AU)
    wbar = 2*np.pi/(P*24*3600)
    sini = np.sin(inc)
    wdot = -5*wbar/sini/sini/((1-e**2)**2)*(k0*r0**5*(Omega0/wbar)**2*(1+M1/M0) + k1*r1**5*(Omega1/wbar)**2*(1+M0/M1))
    wdot *= 180/np.pi*3600*3600*24*365.25
    return wdot

def wdot_spin(M0,M1,e,P,R0,R1,a,Omega0,Omega1,k0,k1,inc,cosa1,cosa2,cosb1,cosb2):
    '''
    Calculate the maximum retrograde rate of apsidal precession of an orbit due to 
    misaligned spins given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e: eccentricity of inner orbit
        P: orbital period of inner orbit (days)
        k0: apsidal constant for the primary body
        k1: apsidal constant for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        inc: inclination of the orbit relative to the observer
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
        a: semi-major axis of orbit (AU)
    Returns rate of precession in arcsec/yr
    '''
    r0 = R0*Rsun/(a*AU)
    r1 = R1*Rsun/(a*AU)
    wbar = 2*np.pi/(P*24*3600)
    sini = np.sin(inc)
    cosi = np.cos(inc)
    trig1 = cosa1*(cosa1-cosb1*cosi) + 0.5*sini**2*(1-5*cosa1**2)
    trig2 = cosa2*(cosa2-cosb2*cosi) + 0.5*sini**2*(1-5*cosa2**2)
    wdot = -wbar/sini/sini/((1-e**2)**2)*(k0*r0**5*(Omega0/wbar)**2*(1+M1/M0)*trig1 + k1*r1**5*(Omega1/wbar)**2*(1+M0/M1)*trig2)
    wdot *= 180/np.pi*3600*3600*24*365.25
    return wdot

def Wdot_spin(M0,M1,e,P,R0,R1,a,Omega0,Omega1,k0,k1,inc,cosa1,cosa2,cosb1,cosb2):
    '''
    Calculate the maximum retrograde rate of apsidal precession of an orbit due to 
    misaligned spins given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e: eccentricity of inner orbit
        P: orbital period of inner orbit (days)
        k0: apsidal constant for the primary body
        k1: apsidal constant for the secondary body
        Omega0: rotation rate of primary body (rad/s)
        Omega1: rotation rate of secondary body (rad/s)
        inc: inclination of the orbit relative to the observer
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
        a: semi-major axis of orbit (AU)
    Returns rate of precession in arcsec/yr
    '''
    r0 = R0*Rsun/(a*AU)
    r1 = R1*Rsun/(a*AU)
    wbar = 2*np.pi/(P*24*3600)
    sini = np.sin(inc)
    cosi = np.cos(inc)
    trig1 = cosa1*(cosb1-cosa1*cosi)
    trig2 = cosa2*(cosb2-cosa2*cosi)
    Wdot = -wbar/sini/sini/((1-e**2)**2)*(k0*r0**5*(Omega0/wbar)**2*(1+M1/M0)*trig1 + k1*r1**5*(Omega1/wbar)**2*(1+M0/M1)*trig2)
    Wdot *= 180/np.pi*3600*3600*24*365.25
    return Wdot
    

def wdot_GR(M0,M1,e1,P1):
    '''
    Calculate the rate of apsidal precession of an orbit due to GR given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e1: eccentricity of inner orbit
        P1: orbital period of inner orbit (days)
    Returns rate of precession in arcsec/yr
    '''
    
    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    
    Cg = 3*G**2*m0*m1*(m0+m1)/(a1**2*c**2)
    
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = Cg/G1/((1-e1**2)**(1/2))
    wdot *= 3600*24*365.25*180*3600/np.pi
    
    return wdot

def wdot_Tid(M0,M1,e1,P1,k0,k1,R0,R1):
    '''
    Caluclate rate of apsidal precession of an orbit due to tides given:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        e: eccentricity of inner orbit
        P: orbital period of inner orbit (days)
        k0: second Love number for the primary body
        k1: second Love number for the secondary body
        R0: Radius of primary (R_sun)
        R1: Radius of secondary (R_sun)
    '''
    a1 = sma_tot(P1,M0,M1)*AU
    
    m0 = M0*Msun
    m1 = M1*Msun
    r0 = R0*Rsun
    r1 = R1*Rsun
    
    Ct0 = 0.5*k0*G*m1**2*r0**5/(a1**6)
    Ct1 = 0.5*k1*G*m0**2*r1**5/(a1**6)
    
    fce = (1 + (3*e1**2)/2 + (e1**4)/8)*(1-e1**2)**(-9/2)
    G1 = m0*m1/(m0+m1)*(G*(m0+m1)*a1*(1-e1**2))**(1/2)
    
    wdot = 15*(Ct0 + Ct1)*fce/G1
    wdot *= 3600*24*365.25*180*3600/np.pi
    
    return wdot

def wdot_Tid_ragozzine(M1,M2,e,P,k2_1,k2_2,R1,R2):
    '''
    Caluclate precession from tides (taken from Ragozzine & Wolf 2009) give:
        M0: Mass of primary (M_sun)
        M1: Mass of secondary (M_sun)
        R1: Radius of primary (R_sun)
        R2: Radius of secondary (R_sun)
        e: eccentricity of inner orbit
        k2_1: second Love number for the primary body
        k2_2: second Love number for the secondary body
        P: orbital period of inner orbit (days)
    '''
    n = 2*np.pi*365.25/P
    a = sma_tot(P, M1,M2)*AU/Rsun
    f2e = (1-e**2)**(-5)*(1+3/2*e**2+1/8*e**4)
    wdottid1 = 7.5*k2_1*(R1/a)**5*M2/M1*f2e*n
    wdottid2 = 7.5*k2_2*(R2/a)**5*M1/M2*f2e*n
    
    return (wdottid1 + wdottid2)*3600*180/np.pi

def planet_mass(M0,M1,e1,e2,P1,P2,w1,inc,wdot,tides=False,k2_0=0,k2_1=0,R0=1,R1=1):
    

    wdot -= wdot_GR(M0,M1,e1,P1)
    if tides:
        wdot -= wdot_Tid(M0, M1, e1, P1, k2_0, k2_1, R0, R1)
    if wdot < 0:
        wdot = 0.1
    
    m0 = 0.0000000000000000001
    m1 = 10000
    wd0 = wdot_dyn(M0,M1,m0,e1,e2,P1,P2,w1,inc)
    wd1 = wdot_dyn(M0,M1,m1,e1,e2,P1,P2,w1,inc)

    if wd0>wdot:
        x=0
        while x==0:
            m1 = m0
            m0 = m0/1000
            wd0 = wdot_dyn(M0,M1,m0,e1,e2,P1,P2,w1,inc)
            if wd0<wdot:
                x=1


    Dwd = wd1 - wd0
    i = 0
    while abs(Dwd) >= 0.001:
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

def planet_mass_scp(M0,M1,e1,e2,P1,P2,w1,inc,wdot):
    '''
    Root finder for the Mass of a body from precession of inner binary, given:
        M0: Central mass (M_sun)
        K: semi-amplitude (m/s)
        P: Period (days)
        e: eccentricity
    Uses scipy.optimize.root
    '''
    M0,M1,e1,e2,P1,P2,w1,inc,wdot = np.atleast_1d(M0,M1,e1,e2,P1,P2,w1,inc,wdot)
    if len(e2)<len(M0):
        e2 = np.ones(len(M0))*e2[0]
    if len(e1)<len(M0):
        e1 = np.ones(len(M0))*e1[0]
    if len(inc)<len(M0):
        inc = np.ones(len(M0))*inc[0]
    if len(P2)<len(M0):
        P2 = np.ones(len(M0))*P2[0]

    masses=[]
    for i in range(len(M0)):
        def fun(m0):
            xxx = wdot_dyn(M0[i],M1[i],m0,e1[i],e2[i],P1[i],P2[i],w1[i],inc[i]) - wdot[i] + wdot_GR(M0[i],M1[i],e1[i],P1[i])
            # print(xxx)
            return xxx
        Mass = sp.optimize.root(fun,0.01)
        masses.append(Mass.x[0])
        # print(Mass.x)
    if Mass.message != 'The solution converged.':
        print('Warning: Mass not conerged with message',Mass.message)
        
    if len(masses)>=2:
        ret = masses
    else:
        ret = masses[0]
    return ret

def delT(a,e,w0,wdot,t):
    '''
    Calculates change in eclipse time from LTTE relative to a linear ephemeris given:
        a: semi-major axis of the eclipsing body about the centre of mass(AU)
        e: eccentricity
        w0: initial argument of pericentre (of the eclipsing body) rad
        wdot: apsidal precession rate (arcsec/yr)
        t: time after initialisation at which we are measuring (days)
    returns dT in seconds
    '''
    
    dw = t*wdot/365.25/3600*np.pi/180 #dw in radians
    
    return a*AU*(1-e**2)/c*(1/(1-e*np.sin(w0)) - 1/(1-e*np.sin(w0+dw)))

def delTs_from_wdot(M0,M1,P,e,w0,wdot,t):
    '''
    Calculates change in eclipse time from LTTE relative to a linear ephemeris given:
        M0: Mass of primary (Msun)
        M1: Mass of secondary (Msun)
        P: period of binary (days)
        e: eccentricity
        w0: initial argument of pericentre (of the eclipsing body) rad
        wdot: apsidal precession rate (arcsec/yr)
        t: time after initialisation at which we are measuring (days)
    returns change in eclipse time for the primary eclipse and secondary eclipse in seconds
    '''
    a0,a1 = sma_2body(P, M0, M1)
    
    dT_pri = delT(a1, e, w0, wdot, t)
    dT_sec = delT(a0, e, w0-np.pi, wdot, t)
    
    return dT_pri,dT_sec

def X_parameter(e,w):
    return np.pi + 2*np.arctan(e*np.cos(w)/((1-e**2)**(1/2)))

def w_from_X(e,X):
    return np.arccos((1-e**2)**(1/2)/e*np.tan((X-np.pi)/2))

def w_from_ts(tpri,tsec,P,e):
    
    c = 2*np.pi*(tsec-tpri)/P
    
    def fun(x):
        return x - np.sin(x) - c
    X = sp.optimize.root(fun,np.pi)
    
    w = w_from_X(e, X[0])
    
    return(w)

def tsec(tpri,e,w,P):
    X = X_parameter(e, w)
    return P*(X - np.sin(X))/2/np.pi + tpri

def tpri(tsec,e,w,P):
    X = X_parameter(e, w)
    return -P*(X - np.sin(X))/2/np.pi + tsec

def wdot_tpri_diff(w0,wdot,time,P,e):
    '''
    Calculate change in offset between primary and secondary eclipses due to apsidal precession of the binary given:
        w0: initial w (rad)
        wdot: precession rate (arcsec/yr)
        time: timespan between measurements
        P: period of the binary (days)
        e: eccentricity of the binary 
    returns offset in seconds
    '''
    w1 = w0 + time*wdot*asyr_to_radday
    
    tpri1 = tpri(0,e,w0,P)
    tpri2 = tpri(0,e,w1,P)
    
    return (tpri1 - tpri2)*24*3600

def t_offset(P,e,w):
    '''
    Offset between primary and secondary eclipse times, to first order in e
    '''
    return P/2*(1 + 4*e*np.cos(w)/np.pi)

def delt_wdot(P,e,w0,wdot,time):
    '''
    change in offset between primary and secondary eclipse, due to apsidal precession
    '''
    w1 = w0 + time*wdot*asyr_to_radday

    off1 = t_offset(P,e,w0)*24*3600
    off2 = t_offset(P,e,w1)*24*3600

    return off2 - off1


#################################################
#astrometry functions
#################################################

def vtan(M0,M1,P):
    a0,a1=sma_2body(P,M0,M1)
    v_circ = ((M0+M1)*Msun*G/a1/AU)**(1/2)
    return v_circ*M1/M0

def vtan2(M0,M1,atot):
    a1 = atot*M0/(M0+M1)
    v_circ = ((M0+M1)*Msun*G/a1/AU)**(1/2)
    return v_circ*M1/M0

def vtan2_ecc(M0,M1,atot):
    a1 = atot*M0/(M0+M1)
    v_circ = ((M0+M1)*Msun*G/a1/AU)**(1/2)
    return v_circ*M1/M0

def vtan_ecc(M0,M1,P,ecc,f):
    '''
    Calculate tangential velocity caused and instantaneus time where
        f = true anomaly
    by an orbiting body with
        M1 = mass (M_sun)
        ecc = eccentricity
        P = orbital period
    around a central body with 
        M0 = mass (M_sun)
    '''
    a = sma_tot(P,M0,M1)*AU
    return (G*(M1*Msun)**2/(M0*Msun+M1*Msun))**(1/2)*(2*(1+ecc*np.cos(f))/a/(1-ecc**2) - 1/a)**(1/2)

def Thiele_Innes(a0,w,W,cosi):
    A = a0*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*cosi)
    B = a0*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*cosi)
    F = -a0*(np.sin(w)*np.cos(W) + np.cos(w)*np.sin(W)*cosi)
    G = -a0*(np.sin(w)*np.sin(W) - np.cos(w)*np.cos(W)*cosi)
    
    return A,B,F,G

def wss(da,dd,par,mua,mud,t,psi,pf,tref):
    T = t - tref
    return (da + mua*T)*np.sin(psi) + (dd +mud*T)*np.cos(psi) +par*pf

def wk_orb(P,Tper,e,cosi,W,w,a0,t,psi):
    
    A,B,F,G = Thiele_Innes(a0, w, W, cosi)
    X,Y = ellip_rectang(t, P, e, Tper)
    
    return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)
 
def wk_TI(P,Tper,e,A,B,F,G,t,psi):   
    
    X,Y = ellip_rectang(t, P, e, Tper)
    return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)

def ra_dec_orb(P,Tper,e,cosi,W,w,a0,t):
    
    A,B,F,G = Thiele_Innes(a0, w, W, cosi)
    X,Y = ellip_rectang(t, P, e, Tper)
    
    return B*X + G*Y, A*X + F*Y

def ra_dec_TI(A,B,F,G,P,Tper,e,t):
    X,Y = ellip_rectang(t, P, e, Tper)
    return B*X + G*Y, A*X + F*Y

def Campbell_a(A,B,F,G):
    u = (A**2 + B**2 + F**2 + G**2)/2
    v = A*G - B*F
    a = (u + ((u+v)*(u-v))**(0.5))**(0.5)
    return a

def Campbell_W_w(A,B,F,G):
    X = np.arctan((B-F)/(G+A))
    Y = np.arctan((B+F)/(G-A))
    
    if np.sin(X)*(B-F) <0:
        X += np.pi
        
    if np.sin(Y)*(-B-F) <0:
        Y+=np.pi

    w = (X+Y)/2
    W = (X-Y)/2
    
    
    if W<0:
        W+=np.pi
        w+=np.pi
        
    while W < 0:
        W += np.pi
    
    while W > np.pi:
        W -= np.pi
        
    while w < 0:
        w += 2*np.pi
        
    while w > 2*np.pi:
        w -= 2*np.pi
        
    return W,w
    
def Campbell_i(A,B,F,G,w,W):
    d1 = abs((A+G)*np.cos(w-W))
    d2 = abs((F-B)*np.sin(w-W))
    
    d3 = abs((A-G)*np.cos(w+W))
    d4 = abs((F+B)*np.sin(w+W))
    
    if d1>=d2:
        i = 2*np.arctan((d3/d1)**(0.5))
    else:
        i = 2*np.arctan((d4/d2)**(0.5))
        
    return i

def Campbell(A,B,F,G):
    '''
    Taken from "Halbwachs et al. 2023"
    '''
    
    A,B,F,G = np.atleast_1d(A, B, F, G)
    W,w,inc = np.zeros_like(A),np.zeros_like(A),np.zeros_like(A)
    
    a = Campbell_a(A, B, F, G)
    for i in range(len(A)):
        W[i],w[i] = Campbell_W_w(A[i], B[i], F[i], G[i])
        inc[i] = Campbell_i(A[i], B[i], F[i], G[i], w[i], W[i])
    
    return a,w,W,inc

def astrometric_SNR(M0,M1,P,dist,Beta=0,Nobs=70,sigma=30):
    '''
    Gives the signal-noise ratio of an astrometric orbit as detectable in Gaia for a binary/exoplanetary orbit. Following the equations in Sahlmann, Triaud, and Martin 2015.
        M0: Mass of primary body (Msun)
        M1: Mass of secondary body (Msun)
        P: Orbital period (days)
        dist: distance to system (pc)
        Beta: flux ratio F1/(F1+F0) assumed zero for SB1/exoplanet
        Nobs: number of Gaia observations (default 70)
        sigma: precision of individual epoch measurments (micro-arcseconds) default 30
    '''
    abar_rel = sma_tot(P,M0,M1) #semi-major axis of the orbit of the primary body in AU
    arel = (abar_rel/dist)*10**6 #abar_rel is in AU, arel in micro-arcsec

    f = M1/(M0+M1)
    alpha = arel*(f-Beta)

    SNR = alpha*(Nobs**0.5)/sigma
    return SNR


#################################################
#ETV funtions
#################################################

def semi_amp_LTTE(P,M0,M1,inc=np.pi/2):
    '''
    Calculate semi-amplitude of ETV signal given:
        P: Period (days)
        M0: Mass of eclipsing binery (M_sun)
        M1: Mass of perturbing body (M_sun)
    '''
    M_tot = (M0+M1)*Msun
    m = M1*Msun
    Period = P*24*3600
    
    return m/M_tot/c*(G*M_tot/4/(np.pi**2))**(1/3)*Period**(2/3)

def get_LTTE_mass_scp(K,M0,P):
    '''
    Root finder for the Mass of a body, given:
        M0: Central mass (M_sun)
        K: semi-amplitude (s)
        P: Period (days)
        e: eccentricity
    Uses scipy.optimize.root
    '''
    K,M0,P = np.atleast_1d(K,M0,P)
    
    masses=[]
    for i in range(len(K)):
        def fun(M2):
            return semi_amp_LTTE(P[i],M0[i],M2) - K[i] 
        # print(P[i],M0[i],K[i])
        Mass = sp.optimize.root(fun,M0[i]/4)
        if Mass.message != 'The solution converged.':
            print('Warning: Mass not conerged with message',Mass.message)
        masses.append(Mass.x[0])
    if len(masses)>=2:
        ret = masses
    else:
        ret = masses[0]
    return ret    

def ET_wdot_mass(ephem1,P,M0,M1,e1,P2,w1,e2=0):
    wdot = (360/(P/365.25))*3600    

    return planet_mass(M0,M1,e1,e2,ephem1,P2,w1,0,wdot)

def keplerian_etv(epoch, ref_T, ephem1, p, k, ecc, omega, t0):
    '''
    Obtains the eclipse time variations due to a keplerain orbit for an array of times time, given 
    Period P, semi-amplitude k (in seconds), eccentricity ecc, argument of pericentre omega, time of 
    preicentre passage t0 and systemic velocity vsys.
    Returns the array (same size as time) of radial velocities
    '''
    time = epoch*ephem1 +ref_T
    et = np.zeros_like(time)
    p, k, ecc, omega, t0 = np.atleast_1d(p, k, ecc, omega, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            et += k[i]/(24*3600) * ((1-ecc[i]**2)/(1+ecc[i]*np.cos(nu))*np.sin(omega[i]+nu)+ ecc[i]*np.sin(omega[i])) * (1 - (ecc[i]*np.cos(omega[i]))**2)**(-0.5)
    return et

def ephem_etv(epoch, ref_T, ephem1,  ephem2=0, ephem3=0):
    '''
    Obtains the eclipse time variations from up to a cubic ephemeris given:
        epoch: array of eclipse epochs
        ref_T: reference time for epoch 0
        ephem1,2,3 linear quadratic and cubic terms for the ephemeris (these 
        are the Period of the binary/planet and its first and second derivatives)
    '''
    
    et = np.ones_like(epoch)*ref_T
    et += epoch*ephem1 +(ephem2*ephem1*epoch**2)/2 + (ephem3*ephem1**2*epoch**3)/6
    
    return et

# def precession_etv(epoch, )

#################################################
#one-two punch
#################################################

def P_from_vel(M,v,e,w):
    '''
    Planet period based on orbital velocity given:
        M: total inner mass (M_sun)
        v: measured velocity at transit (m/s)
        e: eccentricity of orbit
        w: argument of pericentre (rad)
    returns:
        P: planet orbital period (days)
    '''
    return (2*np.pi*G*M*Msun/v**3 *((e*np.sin(w) + 1)/(1-e**2)**(0.5))**3)/3600/24

def vel_transit(P,e,Tecl,T1,T2,wbin,M0,M1,R):
    '''
    Calculate velocity planet during a circumbinary transit given:
        P: orbital period of the binary (days)
        e: binary eccentricity
        Tecl: Time of the primary/secondary eclipse (days) primary/secondary depending on which is being transited
        T1: ingress start/end time (days)
        T2: egress start/end time (days) start/end doesn't matter so long as consostent between T1 and T2
        wbin: argument of pericentre of the ***star being transited*** (rad) *make sure this is not the w of the other star
        M0: Mass of transited star (M_sun)
        M1: mass of companion star (M_sun)
        R: Radius of transited star (R_sun)
    Returns:
        vel: velocity (m/s)
    '''

    a=sma_2body(P,M0,M1)[0]*AU
    phi = np.pi/2 - wbin

    Mecl = mean_anom(phi,e)
    
    M1, M2 = 2*np.pi*(T1-Tecl)/P + Mecl, 2*np.pi*(T2-Tecl)/P + Mecl
    E1, E2 = ecc_anomaly(M1,e), ecc_anomaly(M2,e)
    nu1, nu2 = true_anomaly(E1,e), true_anomaly(E2,e)
    
    theta1, theta2 = nu1 - phi, nu2 - phi

    r1 = a*(1-e**2)/(1+e*np.cos(nu1))
    r2 = a*(1-e**2)/(1+e*np.cos(nu2))

    x1 = r1*np.sin(theta1)
    x2 = r2*np.sin(theta2)


    DX = 2*R*Rsun - (x2-x1)
    
    DT = (T2-T1)*24*3600

    vel = DX/DT
    return vel

def onepunch_period(P,ebin,Tecl,T1,T2,wbin,M0,M1,R,epl=0,wpl=0):
    '''
    Estimate of orbital period of circumbinary planet based velocity planet during a single transit given:
        P: orbital period of the binary (days)
        ebin: binary eccentricity
        Tecl: Time of the primary/secondary eclipse (days) primary/secondary depending on which is being transited
        T1: ingress start/end time (days)
        T2: egress start/end time (days) start/end doesn't matter so long as consostent between T1 and T2
        wbin: argument of pericentre of the ***star being transited*** (rad) *make sure this is not the w of the other star
        M0: Mass of transited star (M_sun)
        M1: mass of companion star (M_sun)
        R: Radius of transited star (R_sun)
        epl: planet eccentricity (assumed 0)
        wpl: planet argument of pericentre (assumed 0)
    Returns:
        P: Planet orbital period (days)
    '''

    vel = vel_transit(P,ebin,Tecl,T1,T2,wbin,M0,M1,R)
    P = P_from_vel(M0+M1,vel,epl,wpl)

    return P

def vel_onetwo(P,e,Tecl1,Tecl2,T1,T2,w1,w2,M0,M1):
    '''
    Calculate velocity planet during a circumbinary transit. The 1 and 2 parameters should be ordered in the time order to assess prograde vs retrograde:
        P: orbital period of the binary (days)
        e: binary eccentricity
        Tecl1: Time of the eclipse of the first star (days) 
        Tecl2: Time of the eclipse of the second star (days) 
        T1: mid-transit time of first transit (days)
        T2: mid-transit time of second transit (days)
        w1: argument of pericentre of the first star's orbit(rad) *make sure this is not the w of the other star
        w2: argument of pericentre of the second star's orbit(rad) *make sure this is not the w of the other star
        M0: Mass of first star (M_sun)
        M1: mass of second star (M_sun)
    Returns:
        vel: velocity (m/s)
        prograde: bool, true if prograde orbit
    '''

    a1=sma_2body(P,M0,M1)[0]*AU
    a2=sma_2body(P,M0,M1)[1]*AU

    phi1 = np.pi/2 - w1
    phi2 = np.pi/2 - w2

    Mecl1 = mean_anom(phi1,e)
    Mecl2 = mean_anom(phi2,e)
    
    M1, M2 = 2*np.pi*(T1-Tecl1)/P + Mecl1, 2*np.pi*(T2-Tecl2)/P + Mecl2
    E1, E2 = ecc_anomaly(M1,e), ecc_anomaly(M2,e)
    nu1, nu2 = true_anomaly(E1,e), true_anomaly(E2,e)
    
    theta1, theta2 = nu1 - phi1, nu2 - phi2

    r1 = a1*(1-e**2)/(1+e*np.cos(nu1))
    r2 = a2*(1-e**2)/(1+e*np.cos(nu2))

    x1 = -r1*np.sin(theta1)
    x2 = -r2*np.sin(theta2)

    prog = x2>x1

    DX = abs(x2-x1)
    
    DT = (T2-T1)*24*3600

    vel = DX/DT

    return vel, prog

def onetwopunch_period(P,ebin,Tecl1,Tecl2,T1,T2,w1,w2,M0,M1,epl=0,wpl=0):
    '''
    Estimate of orbital period of circumbinary planet based velocity planet during a one-two punch double transit given:
        P: orbital period of the binary (days)
        e: binary eccentricity
        Tecl1: Time of the eclipse of the first star (days) 
        Tecl2: Time of the eclipse of the second star (days) 
        T1: mid-transit time of first transit (days)
        T2: mid-transit time of second transit (days)
        w1: argument of pericentre of the first star's orbit(rad) *make sure this is not the w of the other star
        w2: argument of pericentre of the second star's orbit(rad) *make sure this is not the w of the other star
        M0: Mass of first star (M_sun)
        M1: mass of second star (M_sun)
        epl: planet eccentricity (assumed 0)
        wpl: planet argument of pericentre (assumed 0)
    Returns:
        P: Planet orbital period (days)
        prograde: bool, true if prograde orbit
    '''

    vel,progs = vel_onetwo(P,ebin,Tecl1,Tecl2,T1,T2,w1,w2,M0,M1)
    P = P_from_vel(M0+M1,vel,epl,wpl)

    return P,progs

#################################################
#other functions
#################################################


def T_stab(M0,M1,a1,a2,e2,P2):
    '''
    Calculate stability timescale analystically following Zhang, Naoz and Will 2023
    Assumes binary has circular orbit, and massless planet
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

def AaM_stab(M0,M1,m2,a1,e2):
    q=m2/(M0+M1)
    C = 2.8
    return C*((1+q)*(1+e2)/(1-e2)**(0.5))**(2/5)*a1

def get_AaM_stab(M0,M1,m2,P1,e2):
    a = sma_tot(P1,M0,M1)
    rstab = AaM_stab(M0,M1,m2,a,e2)
    P_stab = P_from_sma(rstab,M0+M1,m2)
    return P_stab
    

def HW_stab_lim(a,e):
    return 2.278 + 3.824*e - 1.71*e**2

def HW_stab_lim_ext(a,e,mu):
    return 1.6 +5.1*e - 2.22*e**2 + 4.12*mu - 4.27*e*mu - 5.09*mu**2

def get_HW_stab(M0,M1,P,e,ext=True,param='P'):
    '''
    Circumbinary stability criterion (Holman & Wiegert 1999)
        M0: primary mass (Msun)
        M1: Secondary mass (Msun)
        P: orbital period (days)
        e: eccentricity
    returns:
        a: Period/semi major axis of stability limit (days/AU)
    '''
    
    a = sma_tot(P,M0,M1)
    mu = np.minimum(M1/(M0+M1),M0/(M0+M1))
    if ext:
        a_pl = HW_stab_lim_ext(a, e, mu)*a
    else:
        a_pl = HW_stab_lim(a, e)*a
    
    P_pl = P_from_sma(a_pl, M0+M1)
    if param=='P':
        ret = P_pl
    else:
        ret = a_pl
    
    return ret

def K_BBH(m1,m2,m3,pin,pout,I,ecc):
    
    M1 = m1*Msun
    M2 = m2*Msun
    M3 = m3*Msun
    
    M12 = M1+M2
    M123=M12+M3
    
    Pin = pin*24*3600
    Pout = pout*24*3600
    
    V00 = ((2*np.pi*G*(M12)**3)/((M123)**2*Pout))**(1/3)
    
    return M1*M2/(M12**2)*(M123/M12)**(2/3)*(Pin/Pout)**(7/3)*V00*np.sin(I)*(1-ecc)**(-7/2)

def interpolate(xs,ys,value):
    '''
    linear interpolation:
        xs: input function x values (numpy array)
        ys: input function y values (numpy array)
        value: x value(s) to interpolate (int, float, numpy array, list)
    returns:
        interpolated value(s) float or numpy array
    '''
    if isinstance(value,np.ndarray) or isinstance(value,list):
        interps=[]
        for val in value:
            print(val)
            ysides = ys[[sum(val>xs)-1,sum(val>xs)]]
            xsides = xs[[sum(val>xs)-1,sum(val>xs)]]
            interps.append(sum((1-abs(val-xsides))*ysides))
        return(np.array(interps))
    else:
        ysides = ys[[sum(value>xs)-1,sum(value>xs)]]
        xsides = xs[[sum(value>xs)-1,sum(value>xs)]]
        return sum((1-abs(value-xsides))*ysides)

######################################
#Lombe-Scargle
######################################


def get_spectral_window_function(freqs, times):
    """
    Function to calculate the spectral window function of a series
    of observations given an array of test frequencies.
    (Roberts et al. 1987, Dawson & Fabrycky 2018)
    
    Parameters
    ----------
    freqs : array_like
        An array of test frequencies.
    times : array_like
        An array of times at which the observations were taken.
    
    Returns
    -------
    array_like
        The spectral window function of the observations for the given
        test frequencies.
    """
    w_v = np.zeros(freqs.shape)
    for idx, freq in enumerate(freqs):
        w_v[idx] = np.sum(np.exp(-2j * np.pi * freq * times)) / len(times)
    return np.abs(w_v)

def GLS_periodogram(times,rvs,errs,p_range=[0.5,2000],fapLevels=np.array([0.1, 0.01,0.001])):
    
    fbeg, fend = 1/p_range[1], 1/p_range[0]
    
    clp_rv = pyPeriod.Gls((times,rvs,errs),fbeg=fbeg,fend=fend)
    
    # Find the frequency with the highest power
    max_frequency = clp_rv.freq[np.argmax(clp_rv.power)]
    
    plevels = clp_rv.powerLevel(fapLevels)
    
    return clp_rv.freq, clp_rv.power, fapLevels, plevels, max_frequency

