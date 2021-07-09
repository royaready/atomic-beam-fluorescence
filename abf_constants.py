# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:56:12 2020

@author: ready
"""
import numpy as np

# fundamental constants
amu = 1.660539066e-27# kg/amu

# Yb_mass = 2.8883228e-25 # kg

# v_pz = 232.3 # m/s

# v_py = 29.03 # m/s
# v_py = 290.03 # m/s
    
h_Planck = 6.62607015e-34 # J Hz^-1

k_Boltzmann = 1.380649*1e-23 # J/K

c = 2.99792458e8 # speed of light in vacuum. m/s

r_e = 2.8179403262e-15 # classical electron radius. m

# atom transition data
Yb_174_1P1_nu_o = 751.526e12 # Yb 1P1 excitation frequency in Hz
Rb_85_2P1half_nu_o = 377.1073857*1e12 # Hz
Ca_40_1P1_nu_o = 709.078235*1e12

f_a_Yb_1P1 = 1.37 # oscillator strength for Yb 1P1. unitless
f_a_Ca_1P1 = 1.75
f_a_Rb_2P1half = 0.342 # Rb 2P1half.

A_Yb_1P1 = 1.92e8 # einstein A coefficient for Yb 1P1. s^-1
A_Ca_1P1 = 2.20e8
A_Rb_2P1half = 3.60e7 #s^-1

tau_o_Yb_1P1 = 1./A_Yb_1P1
tau_o_Ca_1P1 = 1./A_Ca_1P1
tau_o_Rb_2P1half = 1./A_Rb_2P1half

#photodetector
r_m_Yb = 11.3 # A/W
r_m_Rb = 13.75 # at 795 nm (A/W)
r_m_Ca = 13.6 # at 423 nm (A/W)

g = 5.0e5 # transimpedance gain V/A

vout_max = 4.1 # max APD voltage

det_radius = 0.25e-3 # detector surface radius = 0.25e-3 m
# det_radius = 0.25*2.54e-2 # half inch optic

a_det = np.pi*det_radius**2 # detector surface area = 1.96e-7 m^2

# abf setup geometry
det_y_Yb = 77.4e-3  # distance from origin (center of laser / atom interxn) to
                        # the front surface of the APD, defined to be along y
det_y_Ca = 77.4e-3
det_y_Rb = 95e-3 # m. from Ben's thesis                        

# det_y = 77.4e-3 /2 # if light collecting lens is ~1/2 between the fluorescnce and sensor
# det_y = 39.4e-3 # LB5864 focal length corrected for 400 nm light
                        
nozzle_to_intersxn_Yb = 13.17625e-2 # m. ~ 5 3/16", don't trust more than 
                                 # 3 digits or so. 
nozzle_to_intersxn_Ca = 13.17625e-2
nozzle_to_intersxn_Rb = 35.0e-3 # m. Ben's thesis                                
                                 
# det_dist = np.sqrt(det_y**2+nozzle_to_intersxn**2) # this is the distance from 
                        # the nozzle aperture to the photodetector

# vapor pressure coefficients
vp_coeff_Yb = np.array([9.111,-8111.0,-1.0849,0.0])
vp_coeff_Rb = np.array([4.857,-4215,0.0,0.0])
vp_coeff_Ca = np.array([10.127,-9517.0,-1.4030,0.0])

# laser data
w_Yb = 3.5e-3 # beam radius in m
w_Ca = 3.5e-3
w_Rb = 2.7e-3

fwhm = 5.0e6 # full width half max of laser in Hz

# rho = 0.0 # radius of the atomic beam at the laser interaction region in m
# rho = 0.0 # radial disctance from the laser beam axis in (m)

nozzle_radius_Yb = 0.125/2.0*25.4*1e-3 # inches to meters
nozzle_radius_Ca = 0.125/2.0*25.4*1e-3
nozzle_radius_Rb = 1./2.*1e-3 # meters

nozzle_length_Yb = 0.5*25.4*1e-3 # inches to meters
nozzle_length_Ca = 0.5*25.4*1e-3
nozzle_length_Rb = 10.0*1e-2 # cm to meters

nozzle_ratio_Yb = 2*nozzle_radius_Yb / nozzle_length_Yb# gamma = w/L, oven nozzle geometry
nozzle_ratio_Ca = 2*nozzle_radius_Ca / nozzle_length_Ca
nozzle_ratio_Rb = 2*nozzle_radius_Rb / nozzle_length_Rb

Yb_168_mass = 167.9339
Yb_168_abundance = 0.00126
Yb_168_isotope_shift = 1887.4e6

Yb_170_mass = 169.9348
Yb_170_abundance = 0.03023
Yb_170_isotope_shift = 1192.4e6

Yb_171_mass = 170.9363
Yb_171_abundance = 0.1409
# isotope shift interpolated.
Yb_171_isotope_shift = 862.85e6

Yb_171_a_hf = -214.173e6 # Hz
Yb_171_b_hf = 0.0

Yb_171_1_halves_abundance = Yb_171_abundance*(2./6.)


# abundance. Assuming the states are equally populated.
Yb_171_3_halves_abundance = Yb_171_abundance*(4./6.)


Yb_172_mass = 171.9364
Yb_172_abundance = 0.21754
Yb_172_isotope_shift = 533.3e6

Yb_173_a_hf = 57.682e6
Yb_173_b_hf = 609.065e6

Yb_173_mass = 172.9382
Yb_173_abundance = 0.16103
Yb_173_isotope_shift = 266.65e6

Yb_173_3_halves_abundance = Yb_173_abundance*(4./18.)


Yb_173_5_halves_abundance = Yb_173_abundance*(6./18.)


Yb_173_7_halves_abundance = Yb_173_abundance*(8./18)


Yb_174_mass = 173.9389  # u
Yb_174_abundance =0.31896  # fraction
Yb_174_isotope_shift = 0.0  # Hz

Yb_176_mass = 175.9426
Yb_176_abundance = 0.12887
Yb_176_isotope_shift = -509.3e6

Rb_87_2S1half_a_hf = 3.417341305421e9
Rb_87_2P1half_a_hf = 407.0e6
Rb_87_2S1half_b_hf = 0.0
Rb_87_2P1half_b_hf = 0.0
Rb_85_2S1half_a_hf = 1.01191081e9
Rb_85_2P1half_a_hf = 120.5e6
Rb_85_2S1half_b_hf = 0.0
Rb_85_2P1half_b_hf = 0.0

Rb_87_2P1half_isotope_shift = 77.583e6

Rb_85_mass = 84.911794 # u
Rb_85_abundance = 0.7217

Rb_87_mass = 86.909187 # u
Rb_87_abundance = 0.2783

# =============================================================================
# Rubidium abundances: assuming the magnetic sublevels is not a good approximation,
# it looks like \Del F = \pm 1 is more favored than \Del F = 0
# =============================================================================

Rb_85_abundance_denom = 1/(7+5)
Rb_85_f3_f2_abundance = Rb_85_abundance*5/9*7*Rb_85_abundance_denom
Rb_85_f3_f3_abundance = Rb_85_abundance*4/9*7*Rb_85_abundance_denom
Rb_85_f2_f3_abundance = Rb_85_abundance*7/9*5*Rb_85_abundance_denom
Rb_85_f2_f2_abundance = Rb_85_abundance*2/9*5*Rb_85_abundance_denom

total_85_abundance = Rb_85_f3_f2_abundance + Rb_85_f3_f3_abundance + Rb_85_f2_f3_abundance + Rb_85_f2_f2_abundance


Rb_87_abundance_denom = 1/(5+3)
Rb_87_f2_f1_abundance = Rb_87_abundance*1/2*5*Rb_87_abundance_denom
Rb_87_f2_f2_abundance = Rb_87_abundance*1/2*5*Rb_87_abundance_denom
Rb_87_f1_f2_abundance = Rb_87_abundance*5/6*3*Rb_87_abundance_denom
Rb_87_f1_f1_abundance = Rb_87_abundance*1/6*3*Rb_87_abundance_denom

total_87_abundance = Rb_87_f2_f1_abundance + Rb_87_f2_f2_abundance + Rb_87_f1_f2_abundance + Rb_87_f1_f1_abundance


Ca_40_mass =39.962591
Ca_40_abundance =0.96941
Ca_40_isotope_shift = 0.0

Ca_42_mass =41.958618
Ca_42_abundance =0.00647
Ca_42_isotope_shift = 393.5e6

Ca_43_mass =42.958766
Ca_43_abundance =0.00135
Ca_43_isotope_shift = 611.8e6 
Ca_43_a_hf = -15.46e6
Ca_43_b_hf = -9.7e6

Ca_43_f5half_abundance = 6/24*Ca_43_abundance
Ca_43_f7half_abundance = 8/24*Ca_43_abundance
Ca_43_f9half_abundance = 10/24*Ca_43_abundance

Ca_44_mass =43.955480
Ca_44_abundance =0.02086
Ca_44_isotope_shift = 773.8e6

Ca_46_mass =45.953689
Ca_46_abundance =0.00004
Ca_46_isotope_shift = 1159.8e6

Ca_48_mass = 47.952553
Ca_48_abundance =0.00187
Ca_48_isotope_shift = 1513.0e6

Ca_47_mass =46.95454
Ca_47_abundance = 0.0
Ca_47_isotope_shift = 1348.7e6
Ca_47_a_hf = -16.20e6
Ca_47_b_hf = 4.1e6

Ca_47_f5half_abundance = 6/24*Ca_47_abundance
Ca_47_f7half_abundance = 8/24*Ca_47_abundance
Ca_47_f9half_abundance = 10/24*Ca_47_abundance



# =============================================================================
# Hyperfine calculations
# =============================================================================


def saturation_intensity(frequency, Einstein_coeff):
    
    Isat = np.pi*h_Planck*Einstein_coeff*frequency**3/(3*c**2)/10. #mW/cm^2
    
    return Isat

def k_hf(f, i ,j):
    # total spin, nuclear spin, electric angular momentum
    result = f*(f+1.0) - i*(i+1.0) - j*(j+1.0)
    
    return result

def hf_shift(a_hf,b_hf,f,i,j):
    # hyperfine shift
    
    
        
        
    # else:
    if (i >= 1) & (j >= 1):
        
        result = (0.5*a_hf*k_hf(f,i,j) + 0.25*b_hf*(1.5*k_hf(f,i,j)*(k_hf(f,i,j)+1) 
                  - 2.0*i*(i+1)*j*(j+1))/(i*(2*i-1)*j*(2*j-1)))
        
    else: 
        # if i==0.5:
        
        result = 0.5*a_hf*k_hf(f,i,j)
        
    return result

Yb_1P1_Isat = saturation_intensity(Yb_174_1P1_nu_o,A_Yb_1P1)
# print('Yb_1P1_Isat: %.3e'%Yb_1P1_Isat)
#Na_2P1half_Isat = saturation_intensity(3e17/588.995,0.616e8)
#print('Na_2P1half_Isat: %.3e'%Na_2P1half_Isat)
Rb_2P1half_Isat = saturation_intensity(Rb_85_2P1half_nu_o,A_Rb_2P1half)
# print('Rb_2P1half_Isat: %.3e'%Rb_2P1half_Isat)
Ca_1P1_Isat = saturation_intensity(Ca_40_1P1_nu_o,A_Ca_1P1)
# print('Ca_1P1_Isat: %.3e'%Ca_1P1_Isat)

Ca_43_1P1_f9half_hf_shift = hf_shift(Ca_43_a_hf,Ca_43_b_hf,4.5,3.5,1.0)
# print('%.1e'%Ca_43_1P1_f9half_hf_shift)
Ca_43_1P1_f7half_hf_shift = hf_shift(Ca_43_a_hf,Ca_43_b_hf,3.5,3.5,1.0)
# print('%.1e'%Ca_43_1P1_f7half_hf_shift)
Ca_43_1P1_f5half_hf_shift = hf_shift(Ca_43_a_hf,Ca_43_b_hf,2.5,3.5,1.0)
# print('%.1e'%Ca_43_1P1_f5half_hf_shift)
Ca_47_1P1_f9half_hf_shift = hf_shift(Ca_47_a_hf,Ca_47_b_hf,4.5,3.5,1.0)
# print('%.1e'%Ca_47_1P1_f9half_hf_shift)
Ca_47_1P1_f7half_hf_shift = hf_shift(Ca_47_a_hf,Ca_47_b_hf,3.5,3.5,1.0)
# print('%.1e'%Ca_47_1P1_f7half_hf_shift)
Ca_47_1P1_f5half_hf_shift = hf_shift(Ca_47_a_hf,Ca_47_b_hf,2.5,3.5,1.0)
# print('%.1e'%Ca_47_1P1_f5half_hf_shift)

Ca_40_total_shift = Ca_40_isotope_shift
Ca_42_total_shift = Ca_42_isotope_shift
# print('Ca_42_total_shift: %.5e'%Ca_42_total_shift)
Ca_43_1P1_f9half_total_shift = Ca_43_1P1_f9half_hf_shift + Ca_43_isotope_shift
# print('Ca_43_1P1_f9_half_total_shift: %.5e'%Ca_43_1P1_f9_half_total_shift)
Ca_43_1P1_f7half_total_shift = Ca_43_1P1_f7half_hf_shift + Ca_43_isotope_shift
# print('Ca_43_1P1_f7_half_total_shift: %.5e'%Ca_43_1P1_f7_half_total_shift)
Ca_43_1P1_f5half_total_shift = Ca_43_1P1_f5half_hf_shift + Ca_43_isotope_shift
# print('Ca_43_1P1_f5_half_total_shift: %.5e'%Ca_43_1P1_f5_half_total_shift)
Ca_43_cg_shift = ((Ca_43_f9half_abundance*Ca_43_1P1_f9half_total_shift + 
                   Ca_43_f7half_abundance*Ca_43_1P1_f7half_total_shift + 
                   Ca_43_f5half_abundance*Ca_43_1P1_f5half_total_shift) / 
                  (Ca_43_f5half_abundance + Ca_43_f7half_abundance + Ca_43_f9half_abundance)
                  )
#print('Ca_43_cg_shift: %.5e'%Ca_43_cg_shift) #center of gravity 
Ca_44_total_shift = Ca_44_isotope_shift
# print('Ca_44_total_shift: %.5e'%Ca_44_total_shift)
Ca_46_total_shift = Ca_46_isotope_shift
# print('Ca_46_total_shift: %.5e'%Ca_46_total_shift)
Ca_47_1P1_f9half_total_shift = Ca_47_1P1_f9half_hf_shift + Ca_47_isotope_shift
# print('Ca_47_1P1_f9_half_total_shift: %.5e'%Ca_47_1P1_f9_half_total_shift)
Ca_47_1P1_f7half_total_shift = Ca_47_1P1_f7half_hf_shift + Ca_47_isotope_shift
# print('Ca_47_1P1_f7_half_total_shift: %.5e'%Ca_47_1P1_f7_half_total_shift)
Ca_47_1P1_f5half_total_shift = Ca_47_1P1_f5half_hf_shift + Ca_47_isotope_shift
# print('Ca_47_1P1_f5_half_total_shift: %.5e'%Ca_47_1P1_f5_half_total_shift)
Ca_48_total_shift = Ca_48_isotope_shift

Yb_171_1_halves_hf_shift = hf_shift(Yb_171_a_hf,Yb_171_b_hf,0.5,0.5,1.0)
# print('%.4e'%Yb_171_1_halves_hf_shift)
Yb_171_3_halves_hf_shift = hf_shift(Yb_171_a_hf,Yb_171_b_hf,1.5,0.5,1.0)
# print('%.4e'%Yb_171_3_halves_hf_shift)
Yb_173_3_halves_hf_shift = hf_shift(Yb_173_a_hf,Yb_173_b_hf,1.5,2.5,1.0)
# print('%.4e'%Yb_173_3_halves_hf_shift)
Yb_173_5_halves_hf_shift = hf_shift(Yb_173_a_hf,Yb_173_b_hf,2.5,2.5,1.0)
# print('%.4e'%Yb_173_5_halves_hf_shift)
Yb_173_7_halves_hf_shift = hf_shift(Yb_173_a_hf,Yb_173_b_hf,3.5,2.5,1.0)
# print('%.4e'%Yb_173_7_halves_hf_shift)

Yb_171_1_halves_total_shift = Yb_171_isotope_shift + Yb_171_1_halves_hf_shift
Yb_171_3_halves_total_shift = Yb_171_isotope_shift + Yb_171_3_halves_hf_shift
Yb_173_3_halves_total_shift = Yb_173_isotope_shift + Yb_173_3_halves_hf_shift
Yb_173_5_halves_total_shift = Yb_173_isotope_shift + Yb_173_5_halves_hf_shift
Yb_173_7_halves_total_shift = Yb_173_isotope_shift + Yb_173_7_halves_hf_shift
Yb_174_total_shift = Yb_174_isotope_shift
Yb_168_total_shift = Yb_168_isotope_shift
Yb_170_total_shift = Yb_170_isotope_shift
Yb_172_total_shift = Yb_172_isotope_shift
Yb_176_total_shift = Yb_176_isotope_shift

Rb_87_2S1half_f1_hf_shift = hf_shift(Rb_87_2S1half_a_hf, Rb_87_2S1half_b_hf,1,1.5,0.5)
Rb_87_2S1half_f2_hf_shift = hf_shift(Rb_87_2S1half_a_hf, Rb_87_2S1half_b_hf,2,1.5,0.5)
Rb_87_2P1half_f1_hf_shift = hf_shift(Rb_87_2P1half_a_hf, Rb_87_2P1half_b_hf,1,1.5,0.5)
Rb_87_2P1half_f2_hf_shift = hf_shift(Rb_87_2P1half_a_hf, Rb_87_2P1half_b_hf,2,1.5,0.5)
Rb_85_2S1half_f2_hf_shift = hf_shift(Rb_85_2S1half_a_hf, Rb_85_2S1half_b_hf,2,2.5,0.5)
Rb_85_2S1half_f3_hf_shift = hf_shift(Rb_85_2S1half_a_hf, Rb_85_2S1half_b_hf,3,2.5,0.5)
Rb_85_2P1half_f2_hf_shift = hf_shift(Rb_85_2P1half_a_hf, Rb_85_2P1half_b_hf,2,2.5,0.5)
Rb_85_2P1half_f3_hf_shift = hf_shift(Rb_85_2P1half_a_hf, Rb_85_2P1half_b_hf,3,2.5,0.5)

Rb_85_f2_f2 = (Rb_85_2P1half_f2_hf_shift-Rb_85_2S1half_f2_hf_shift 
                   )
Rb_85_f2_f3 = (Rb_85_2P1half_f3_hf_shift-Rb_85_2S1half_f2_hf_shift 
               )
Rb_85_f3_f2 = Rb_85_2P1half_f2_hf_shift-Rb_85_2S1half_f3_hf_shift
Rb_85_f3_f3 = Rb_85_2P1half_f3_hf_shift-Rb_85_2S1half_f3_hf_shift
Rb_87_f1_f1 = Rb_87_2P1half_f1_hf_shift-Rb_87_2S1half_f1_hf_shift+Rb_87_2P1half_isotope_shift 
Rb_87_f1_f2 = Rb_87_2P1half_f2_hf_shift-Rb_87_2S1half_f1_hf_shift+Rb_87_2P1half_isotope_shift 
Rb_87_f2_f1 = Rb_87_2P1half_f1_hf_shift-Rb_87_2S1half_f2_hf_shift+Rb_87_2P1half_isotope_shift
Rb_87_f2_f2 = Rb_87_2P1half_f2_hf_shift-Rb_87_2S1half_f2_hf_shift+Rb_87_2P1half_isotope_shift
