EllipsoidalColl_MG_LCDM.py

Solving the ellipsoidal collapse ODEs of a top-hat dark matter halo in Hu-Sawicki f(R) gravity, with a LambdaCDM background cosmology.

Collapse ODEs: Equation (43) and (44) of MG_EC.pdf
Initial Condition: Equation (45) and (46)

python2.7

***************************
# Parameters

cosmological parameters:

fR0_abs
 
Omega_m0
 
Omega_L0


top-hat halo parameters:

delta_halo_init: the initial overdensity of the top-hat halo

**User needs to adjust the value of delta_halo_init to ensure that the halo is virializad in present day a0=1**

delta_env_init: the initial overdensity of the environment in which the halo is embedded
 
M_halo_input: halo mass (in the unit of solar mass)
The linear growth function (defined in D_fR_interp()) depends on halo mass. For example, when calculating the case of M_halo_input=1e14, one needs to manually choose the corresponding pre-calculated linear growth function D_fRlinM14 in the line 
    f = interpolate.interp1d(lna_fRlin, D_fRlinM13, kind='cubic')


e, p: ellipticity of the top-hat halo

***************************
# Outputs

Y1_fRpert/Y2_fRpert/Y3_fRpert: the dimensionless comoving lengths (normalized by the initial comoving length, defined in Equation (32)) of three main axes of the ellipsoid

delta_ec: the collapse barrier defined in Equation (47)
# EC_code
