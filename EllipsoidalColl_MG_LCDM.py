# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 00:36:54 2018

@author: chzruan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy import interpolate

#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=16, usetex=False)

a_init = 0.008
a0 = 1.0
lna = np.linspace(np.log(a_init), 0, num=int(10000))
a = np.exp(lna)

# initial conditions
delta_halo_init = 0.015372 
delta_env_init = 0.008
M_halo_input = 1e13

#e_p0 = np.array([0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45])
e_p0 = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
e = e_p0[1]
p = 0.0

fR0_abs = 1e-5
Omega_m0 = 0.24 #Omega_Lambda0 = 1-0.24 = 0.76
Omega_L0 = 1.0 - Omega_m0

print "halo mass = %.1f"%np.log10(M_halo_input)
print "delta_halo_init = %.6f"%delta_halo_init
print "e = %.5f"%e


lna_fRlin, D_fRlinM12, D_fRlinM13, D_fRlinM14, D_fRlinM15, D_fRlinM16, D_fRlinM17, D_lcdm = np.loadtxt("fRlinearD_halomass.txt", unpack=True)

def D_fR_interp(a_input):
    lna_input = np.log(a_input)
    if lna_input < np.min(lna_fRlin):
        lna_input = np.min(lna_fRlin)
    if lna_input > np.max(lna_fRlin):
        lna_input = np.max(lna_fRlin)

    f = interpolate.interp1d(lna_fRlin, D_fRlinM13, kind='cubic')
    return f(lna_input)
    
    
def convert_ep_to_lambdas(delta_halo_init, e, p):
    lambda1 = delta_halo_init/3.0 * (1.0 + 3.0*e + p)
    lambda2 = delta_halo_init/3.0 * (1.0 - 2.0*p)
    lambda3 = delta_halo_init/3.0 * (1.0 - 3.0*e + p)
    return np.array([lambda1, lambda2, lambda3])

lambda1, lambda2, lambda3 = convert_ep_to_lambdas(delta_halo_init, e, p)
# initial conditions over

def yukino(s, Y1, Y2, Y3):
    return 1.0 / np.sqrt((Y1**2 + s) * (Y2**2 + s) * (Y3**2 + s))

def alpha_integral1(s, Y1, Y2, Y3):
    return 1.0/(Y1**2 + s) * yukino(s, Y1, Y2, Y3)

def alpha_integral2(s, Y1, Y2, Y3):
    return 1.0/(Y2**2 + s) * yukino(s, Y1, Y2, Y3)

def alpha_integral3(s, Y1, Y2, Y3):
    return 1.0/(Y3**2 + s) * yukino(s, Y1, Y2, Y3)

def alpha1(Y1, Y2, Y3):
    return Y1*Y2*Y3 * quad(alpha_integral1, 0, np.infty, args=(Y1, Y2, Y3))[0]

def alpha2(Y1, Y2, Y3):
    return Y1*Y2*Y3 * quad(alpha_integral2, 0, np.infty, args=(Y1, Y2, Y3))[0]

def alpha3(Y1, Y2, Y3):
    return Y1*Y2*Y3 * quad(alpha_integral3, 0, np.infty, args=(Y1, Y2, Y3))[0]


def Delta_halo(Y1, Y2, Y3):
    return (1.0-lambda1)*(1.0-lambda2)*(1.0-lambda3)/(Y1*Y2*Y3) * (1.0+delta_halo_init) - 1.0

delta_ec_vir = 179.0
Y_vir = ((1.0 - lambda1)*(1.0 - lambda2)*(1.0 - lambda3)*(1.0 + delta_halo_init) / (1.0 + delta_ec_vir))**(1.0/3.0)


def ForceIncrement(a_input, y_env, Y_h1, Y_h2, Y_h3):
    c_div_H0R = 3e3 * (1.12 * np.pi / 3.0)**(1.0/3.0) *  Omega_m0**(1.0/3.0) * (1.0 + delta_halo_init)**(1.0/3.0) * (M_halo_input/1e12)**(-1.0/3.0)  # initial comoving radius
    
    fac1 = ( (1.0 + 4.0*Omega_L0/Omega_m0) / (y_env**(-3) + 4.0 * Omega_L0/Omega_m0 * a_input**3) )**2
    fac2 = ((1.0 + 4.0*Omega_L0/Omega_m0) / ((Y_h1*Y_h2*Y_h3)**(-1) + 4.0 * Omega_L0/Omega_m0 * (a_input/a0)**3))**2
    DeltaR_div_RTH = fR0_abs / Omega_m0 * (a_input)**7.0 * (c_div_H0R**2) * (Y_h1*Y_h2*Y_h3)**(1.0/3.0) * (fac1 - fac2)
    jkl = 3.0*DeltaR_div_RTH - 3.0*DeltaR_div_RTH**2 + DeltaR_div_RTH**3
    if jkl < 1:
        FFF = 1.0/3.0 * jkl
    else:
        FFF = 1.0/3.0
    return FFF



def saki(a_input):
    jkl = a_input**3 * (Omega_m0*(a0/a_input)**3 + 1.0-Omega_m0)**(1.5)
    return 1.0 / jkl

def D_LCDM(a_input):
    return 2.5 * Omega_m0 * np.sqrt(Omega_m0*(1.0/a_input)**3 + 1.0-Omega_m0) * quad(saki, 1e-5, a_input)[0]

def Esquare(a_input):
    return Omega_m0*(1.0/a_input)**3 + 1 - Omega_m0
    

def Omega_m_a(a_input):
    noshiro = Omega_m0 / a_input**3 / Esquare(a_input)
    return noshiro


#############################################
def ellip_coll_fRlinearPert_func(Ys, lna):
    Y1, Y2, Y3, Y4, Y5, Y6, y_env, y_env_prime = Ys
    a = np.exp(lna)
    Force_incre_nom = ForceIncrement(a, y_env, Y1, Y2, Y3)
    Force_incre_froz1 = ForceIncrement(a, y_env, Y_vir, Y2, Y3)
    Force_incre_froz2 = ForceIncrement(a, y_env, Y_vir, Y_vir, Y3)

    dY4_dlna_nom = -(2 - 1.5*Omega_m_a(a))*Y4 - (1+Force_incre_nom) * 1.5 * Omega_m_a(a) * Y1 * (0.5 * alpha1(Y1, Y2, Y3) * Delta_halo(Y1, Y2, Y3) + D_fR_interp(a)/a_init*(lambda1 - 1.0/3.0*delta_halo_init))
    dY5_dlna_nom = -(2 - 1.5*Omega_m_a(a))*Y5 - (1+Force_incre_nom) * 1.5 * Omega_m_a(a) * Y2 * (0.5 * alpha2(Y1, Y2, Y3) * Delta_halo(Y1, Y2, Y3) + D_fR_interp(a)/a_init*(lambda2 - 1.0/3.0*delta_halo_init))
    dY6_dlna_nom = -(2 - 1.5*Omega_m_a(a))*Y6 - (1+Force_incre_nom) * 1.5 * Omega_m_a(a) * Y3 * (0.5 * alpha3(Y1, Y2, Y3) * Delta_halo(Y1, Y2, Y3) + D_fR_interp(a)/a_init*(lambda3 - 1.0/3.0*delta_halo_init))
    
    dY5_dlna_froz1 = -(2 - 1.5*Omega_m_a(a))*Y5 - (1+Force_incre_froz1) * 1.5 * Omega_m_a(a) * Y2 * (0.5 * alpha2(Y_vir, Y2, Y3) * Delta_halo(Y_vir, Y2, Y3) + D_fR_interp(a)/a_init*(lambda2 - 1.0/3.0*delta_halo_init))
    dY6_dlna_froz1 = -(2 - 1.5*Omega_m_a(a))*Y6 - (1+Force_incre_froz1) * 1.5 * Omega_m_a(a) * Y3 * (0.5 * alpha3(Y_vir, Y2, Y3) * Delta_halo(Y_vir, Y2, Y3) + D_fR_interp(a)/a_init*(lambda3 - 1.0/3.0*delta_halo_init))

   
    dY6_dlna_froz2 = -(2 - 1.5*Omega_m_a(a))*Y6 - (1+Force_incre_froz2) * 1.5 * Omega_m_a(a) * Y3 * (0.5 * alpha3(Y_vir, Y_vir, Y3) * Delta_halo(Y_vir, Y_vir, Y3) + D_fR_interp(a)/a_init*(lambda3 - 1.0/3.0*delta_halo_init))

    
    y_env_doubleprime = -0.5*y_env_prime - 0.5*(y_env**(-3) - 1.0) * y_env

    if Y1 > Y_vir:
        return np.array([Y4, Y5, Y6, dY4_dlna_nom, dY5_dlna_nom, dY6_dlna_nom, y_env_prime, y_env_doubleprime])

    if (Y1 <= Y_vir) & (Y2 > Y_vir):
        # Y1 has been frozen
        # Y2 and Y3 are normal
        return np.array([0, Y5, Y6, 0, dY5_dlna_froz1, dY6_dlna_froz1, y_env_prime, y_env_doubleprime])

    if (Y1 <= Y_vir) & (Y2 <= Y_vir) & (Y3 > Y_vir):
        # Y1 and Y2 have been frozen
        # Y3 is normal
        return np.array([0, 0, Y6, 0, 0, dY6_dlna_froz2, y_env_prime, y_env_doubleprime])

    if (Y1 <= Y_vir) & (Y2 <= Y_vir) & (Y3 <= Y_vir):
        # Y1, Y2 and Y3 have been frozen.
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])


track_fRlinearPert = odeint(ellip_coll_fRlinearPert_func, (1.0-lambda1, 1.0-lambda2, 1.0-lambda3, -lambda1, -lambda2, -lambda3, 1.0-delta_env_init/3.0, -delta_env_init/3.0), lna)

#np.savetxt('MG_p'+np.str(p)+'e'+np.str(e)+'_track.txt', track1)
Y1_fRpert = track_fRlinearPert[:, 0]
Y2_fRpert = track_fRlinearPert[:, 1]
Y3_fRpert = track_fRlinearPert[:, 2]
#############################################

# plot
banana = np.zeros(len(a))
banana += Y_vir
plt.figure(figsize=(5*2.5, 5*2.5))
plt.plot(a, Y1_fRpert, label="$X_1 / X_{\\mathrm{init}}$ (fRlin)")
plt.plot(a, Y2_fRpert, label="$X_2 / X_{\\mathrm{init}}$ (fRlin)")
plt.plot(a, Y3_fRpert, label="$X_3 / X_{\\mathrm{init}}$ (fRlin)")

plt.plot(a, banana, '--', label='$X_{\\mathrm{vir}} / X_{\\mathrm{init}}$')
plt.xlabel('$a$', fontsize=24)
plt.ylabel("comoving size", fontsize=24)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(fontsize=24)
plt.savefig("EC_fR.pdf")


delta_ec = D_fR_interp(1.0) / D_fR_interp(a_init)  * delta_halo_init
print "delta_ec = %.4f (fR)"%delta_ec

# delta_env_ec = D_fR_interp(1.0) / D_fR_interp(a_init)  * delta_env_init
# print "delta_env_ec = %.4f"%delta_env_ec
