# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:43:17 2020

@author: ready
"""
import numpy as np
import abf_constants as abc
import glob, os
import math
import matplotlib.pyplot as plt

def load_plot_settings(single_column):
    
# =============================================================================
# from https://matplotlib.org/stable/tutorials/text/usetex.html
# font.family        : serif
# font.serif         : Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
# font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans serif
# font.cursive       : Zapf Chancery
# font.monospace     : Courier, Computer Modern Typewriter


# text.usetex        : true
# =============================================================================
    
    if single_column ==1:
        # plot_dims = [5.75,3.44] # for single columns
        plot_dims = [6.,3.57] # for single columns
    else:
        plot_dims = [2.875,2.15625] # for double columns
    
    font_size = 11    
    
    
    font_family = "serif"
    font_serif = ["Palatino"]
    
    font = {'family' : font_family,
            'weight' : 'bold',
            'size'   : font_size}
    
    plt.rc('font', **font)
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": font_serif,
    "font.size": font_size
    })
    
    return plot_dims,font_size

def load_atom_data(species,oven_temp = None,laser_radius = None, laser_power = None, nozzle_ratio = None ):
    #oven temp in degC, laser_radius in m , laser_power in W , nozzle_ratio 2*radius / length
    
    if species == "Yb":
        
        # atomic properties
        nu_o = abc.Yb_174_1P1_nu_o
        mass = (abc.Yb_168_mass*abc.Yb_168_abundance + abc.Yb_170_mass*abc.Yb_170_abundance + 
                abc.Yb_171_mass*abc.Yb_171_abundance + abc.Yb_172_mass*abc.Yb_172_abundance +
                abc.Yb_173_mass*abc.Yb_173_abundance + abc.Yb_174_mass*abc.Yb_174_abundance +
                abc.Yb_176_mass*abc.Yb_176_abundance)*abc.amu # to kg
        
        A_Einstein = abc.A_Yb_1P1
        tau_o = abc.tau_o_Yb_1P1
        f_a = abc.f_a_Yb_1P1
        
        # vapor pressure coefficients
        vp_coeff = abc.vp_coeff_Yb
        
        #geometry
        nozzle_radius = abc.nozzle_radius_Yb
        nozzle_to_intersxn = abc.nozzle_to_intersxn_Yb
        det_y = abc.det_y_Yb
        det_radius = abc.det_radius
        
        if nozzle_ratio == None:
            nozzle_ratio = abc.nozzle_ratio_Yb
            
        
        # photodetector settings
        r_m = abc.r_m_Yb
        
                
        if oven_temp == None:
            oven_temp = 300.0 #C
            
        
        if laser_radius == None:
            laser_radius = abc.w_Yb
        
        
        
        if laser_power == None:
            laser_power = 10e-3

        
    elif species == "Rb":
        
        # atomic properties
        nu_o = abc.Rb_85_2P1half_nu_o
        mass = (abc.Rb_85_mass*abc.Rb_85_abundance + abc.Rb_87_mass*abc.Rb_87_abundance)*abc.amu # to kg
        A_Einstein = abc.A_Rb_2P1half
        tau_o = abc.tau_o_Rb_2P1half
        f_a = abc.f_a_Rb_2P1half
        
        # vapor pressure coefficients
        vp_coeff = abc.vp_coeff_Rb
        
        #geometry
        nozzle_radius = abc.nozzle_radius_Rb
        nozzle_to_intersxn = abc.nozzle_to_intersxn_Rb
        det_y = abc.det_y_Rb
        det_radius = abc.det_radius
        
        if nozzle_ratio == None:
            nozzle_ratio = abc.nozzle_ratio_Rb
            
        # photodetector settings
        r_m = abc.r_m_Rb
        
        if oven_temp == None:
            oven_temp = 100.0 #C
            
        if laser_radius == None:
            laser_radius = abc.w_Rb
        
        if laser_power == None:
            laser_power = 50e-6
            
    elif species == "Ca":
        
        # atomic properties
        nu_o = abc.Ca_40_1P1_nu_o
        mass = (abc.Ca_40_mass*abc.Ca_40_abundance + abc.Ca_42_mass*abc.Ca_42_abundance + 
                abc.Ca_43_mass*abc.Ca_43_abundance + abc.Ca_44_mass*abc.Ca_44_abundance + 
                abc.Ca_46_mass*abc.Ca_46_abundance + abc.Ca_48_mass*abc.Ca_48_abundance
                )*abc.amu # to kg
        
        A_Einstein = abc.A_Ca_1P1
        tau_o = abc.tau_o_Ca_1P1
        f_a = abc.f_a_Ca_1P1
        
        # vapor pressure coefficients
        vp_coeff = abc.vp_coeff_Ca
        
        #geometry
        nozzle_radius = abc.nozzle_radius_Ca
        nozzle_to_intersxn = abc.nozzle_to_intersxn_Ca
        det_y = abc.det_y_Ca
        det_radius = abc.det_radius
        
        if nozzle_ratio == None:
            nozzle_ratio = abc.nozzle_ratio_Ca
            
        # photodetector settings
        r_m = abc.r_m_Ca
        
        if oven_temp == None:
            oven_temp = 250.0 #C
            
        if laser_radius == None:
            laser_radius = abc.w_Ca
        
        if laser_power == None:
            laser_power = 10e-3
            
        
    
    print('species = %s'%species)
    print('nozzle ratio = %.3e'%nozzle_ratio)
    print('oven temperature = %.0f degC'%oven_temp)
    print('laser radius = %.2e'%laser_radius)
    print('laser power = %.2e W'%laser_power)
    
    return (nu_o, mass, A_Einstein,tau_o,f_a, vp_coeff,nozzle_radius,
            nozzle_to_intersxn,det_y,det_radius,nozzle_ratio,r_m,oven_temp,
            laser_radius,laser_power)

def most_probable_speed(temp,mass):

	# returns the most probable speed. To get units
	# of m/s, provide the temperature in C and 
	# the mass in kg.

    temp_K = temp + 273.15
    
    result = np.sqrt(2*abc.k_Boltzmann*temp_K/mass);
    
    return result

def jm_dist_elwise(gamma,theta):
    # atomic angular distribution (not a probability distribution) from Olander
    # This equation throws an arccos warning despite setting the condition that
    # only noodle_param < 1.0 are used for arccos. I've looked at the function
    # output in a text file and I plotted the data and the function appears to
    # work exactly as intended, no idea why it's throwing a warning.
    result = None
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    zeta_0 = (0.5-1./(3*gamma**2)*(1-2*gamma**3+(2*gamma**2-1)*np.sqrt(1+gamma**2))/(
                (np.sqrt(1+gamma**2)-gamma**2*np.arcsinh(1/gamma))
                )
        )
    zeta_1 = (1.0 - zeta_0)
    
    cos_theta_gamma = gamma*cos_theta
    p = (sin_theta/cos_theta_gamma)
          
    result = np.where(p>=1.,(
        zeta_0*cos_theta + 4.0*gamma/(3*np.pi)*(zeta_1-zeta_0)*(cos_theta)**2/sin_theta
        ),(
           zeta_0*cos_theta + (2/np.pi)*cos_theta*(
           (1.0-zeta_0)*(np.arccos(p)-p*np.sqrt(1.0-(p)**2))+(2./3.)*(zeta_1-zeta_0)*(1.0-(1.0-(p)**2)**(3/2))/p
           )
          )
          )
                   
    return result

def solid_angle_circular_detector_elwise(det_rad,det_y,det_z,r_x,r_y,r_z):
    # det_rad (m) is the radius of the detector
    # delta (m) is the infinitesimal element size in the solid angle integral

    d_omega_list = np.array([]) 
    
    #number of infinitesimal elements
    delta_steps = 21.
    
    true_area = np.pi*det_rad**2

    X, Z = np.mgrid[-det_rad:det_rad:delta_steps*1j,-det_rad:det_rad:delta_steps*1j]
    
    x = X.flatten()
    z = Z.flatten()
    
    delta = x[int(delta_steps)]-x[0]
    
    distance_now = np.sqrt((x[0]- r_x)**2 + (det_y - r_y)**2 + (det_z + z[0] - r_z)**2)

    d_omega_now = delta**2*(det_y - r_y)/distance_now**3

    # keep in mind this method only works if x and z is of length 2 or greater
    d_omega_list = np.append(d_omega_list,d_omega_now)
    
    distance_now = np.sqrt((x[1]- r_x)**2 + (det_y - r_y)**2 + (det_z + z[1] - r_z)**2)

    d_omega_now = delta**2*(det_y - r_y)/distance_now**3

    d_omega_list = np.append([d_omega_list],[d_omega_now],axis=0)

    for i in range(2,len(x)):
            
        distance_now = np.sqrt((x[i]- r_x)**2 + (det_y - r_y)**2 + (det_z + z[i] - r_z)**2)

        d_omega_now = delta**2*(det_y - r_y)/distance_now**3

        d_omega_list = np.append(d_omega_list,[d_omega_now],axis = 0)

    good_status_indices = np.where(np.less_equal(np.sqrt((np.absolute(x)-.5*delta)**2+(np.absolute(z)-.5*delta)**2),det_rad))
    
    # now only select the infinitesimal integral bits that are in the circle
    d_omega_list_in_circle = d_omega_list[good_status_indices[0][:]]

    d_omega_list_in_circle_transpose = np.transpose(d_omega_list_in_circle)
    
    omega = np.sum(d_omega_list_in_circle_transpose,axis=1)
    
    
    integrated_area = len(good_status_indices[0][:])*delta**2
    
    fraction_true_area = integrated_area/true_area
   
    normalized_solid_angle = omega/fraction_true_area
    
    return normalized_solid_angle

def atom_flux_elwise(Yb_oven_temp_C,ang_distribution,nozzle_radius,
              interaction_distance,mass,vp_coeff,speed_z):
    
    # polar angle in rad
    
    o_num_density = oven_number_density(vp_coeff,Yb_oven_temp_C)
    
    atom_exit_rate = oven_atom_exit_rate(o_num_density,speed_z,nozzle_radius)
    
    result = atom_exit_rate*ang_distribution/(interaction_distance**2)
    
    return result

def oven_number_density(vp_coeff,Yb_oven_temp_C):
    
    Yb_oven_temp_K = Yb_oven_temp_C + 273.15
    
    
    Yb_oven_P = vapor_pressure(vp_coeff,Yb_oven_temp_C) #Pa
    
    result=Yb_oven_P/(abc.k_Boltzmann*Yb_oven_temp_K)
    
    return result

def oven_atom_exit_rate(o_num_density,speed,nozzle_radius):
    
    oven_cs_area = np.pi*nozzle_radius**2
    
    result = o_num_density*speed* oven_cs_area/(4*np.pi)
    # print('oven atom exit rate is %.3e atoms/s'%result)
    
    return result

def vapor_pressure(vp_coeff,temp_C):
    
    vap_a = vp_coeff[0]
    vap_b = vp_coeff[1]
    vap_c = vp_coeff[2]
    vap_d = vp_coeff[3]
    
    temp_K = temp_C + 273.15 # Celsius to Kelvin
    
    logP = 5.006+vap_a+vap_b/(temp_K)+vap_c*np.log10(temp_K)+vap_d/(temp_K**3)
    
    P = 10**logP
    
    return P

def excitation_rate(nu_gamma_now,nu_o,A_Einstein,mass,oven_temp,cos_alpha,beam_power,laser_pos,beam_size,f_a):
    # mass in kg, oven temp in C
    oven_temp_K = oven_temp + 273.15
    # mpv= most_probable_velocity(oven_temp,mass)
    mpv= most_probable_speed(oven_temp,mass)
    absolute_laser_freq = nu_gamma_now + nu_o
        
    
    def s_spec_fn(pos,beam_size):
        #spatial distribution function = fraction of all photons / area
    
        result = 2/(np.pi*beam_size**2)*np.exp(-2*(pos/beam_size)**2)
                    
        return result
    
    # this has length of number of integrand positions
    factor_outside_integral = (beam_power/(abc.h_Planck*absolute_laser_freq)*s_spec_fn(laser_pos,beam_size)*np.pi*
              abc.r_e*abc.c*f_a*2/(np.pi*A_Einstein)
              )
    
    
    # these are the integrand bounds.
    symmetric_int_range = 20.0*abc.fwhm
    int_step_size = abc.fwhm/10.
    nu_low_bound = nu_gamma_now-symmetric_int_range
    nu_high_bound = nu_gamma_now+symmetric_int_range
    
    # want odd number of steps so zero is one of the points
    num_integrand_steps = len(np.arange(0,2*symmetric_int_range,int_step_size))+1 
    num_integrand_steps = num_integrand_steps + np.mod(num_integrand_steps,2)
    num_integrand_steps = num_integrand_steps - 1
    
    nu_integrand_range = np.linspace(nu_low_bound,nu_high_bound,num_integrand_steps)
    
    nu_diff = nu_integrand_range - nu_gamma_now
    
    nu_dependence_factor = (np.pi*A_Einstein/2*np.sqrt(4*np.log(2)/np.pi)/
                            abc.fwhm*np.exp(-4.0*np.log(2)*(nu_diff/abc.fwhm)**2)
                            )
    
    # give nu_dependence_factor dimensions of nu times alpha
    tile_nu_dependence_factor = np.tile(nu_dependence_factor,(len(cos_alpha),1))
    
# =============================================================================
#     here is the integral over all speeds
    
    # speed_range = 550. #m/s
    speed_int_step_size = 5.
    speed_range = np.ceil(3*mpv/speed_int_step_size)*speed_int_step_size #m/s
    
    num_MB_int_steps = len(np.arange(1.0,speed_range,speed_int_step_size))+1
    num_MB_int_steps = int(num_MB_int_steps + np.mod(num_MB_int_steps,2))

    speed_integrand_range = np.linspace(1.0,speed_range,num_MB_int_steps)
    
    MB_factor = (
        np.sqrt(2/np.pi)*(mass/(abc.k_Boltzmann*oven_temp_K))**(3/2)*
        speed_integrand_range**2*np.exp(-mass*speed_integrand_range**2/(2*abc.k_Boltzmann*oven_temp_K))
                  )
    
    
# =============================================================================

    def MB_integrate(nu):
        
        # 2D , # of volume elements by speed steps
        # tile_cosalpha = np.transpose(np.tile(np.cos(alpha),(num_MB_int_steps,1)))
        tile_cosalpha = np.transpose(np.tile(cos_alpha,(num_MB_int_steps,1)))
        
        doppler_term = -nu_o*tile_cosalpha*speed_integrand_range/abc.c
        # print(doppler_term.shape)
        
        # nu is really nu - nu_a. The doppler turn is nu - nu_a(1-cos(alpha)*v/c). 
        # With my convention fo shifting nu by nu_a, the equvialent expression is nu + nu_a*cos(alpha)*v/c
        doppler_diff = nu - doppler_term
        
        Lorentzian_DB_factor = (
            A_Einstein/(2*np.pi*2*np.pi)/((A_Einstein/(2*np.pi))**2/
                                4+(doppler_diff)**2)
                                )
        
        total_speed_factor = MB_factor*Lorentzian_DB_factor
        
        integrate_over_speed = np.trapz(total_speed_factor,x=speed_integrand_range)
        
        return integrate_over_speed
    
    transpose_lineshape_speed_factor_integrated = np.append(
        [MB_integrate(nu_integrand_range[0])],[MB_integrate(nu_integrand_range[1])],axis=0)
    
    for this_nu in nu_integrand_range[2:]:
        transpose_lineshape_speed_factor_integrated = np.append(
            transpose_lineshape_speed_factor_integrated,[MB_integrate(this_nu)],axis=0)
    
    lineshape_speed_factor_integrated = np.transpose(transpose_lineshape_speed_factor_integrated)
    
    #now multiply by the the local photon flux
    lineshape_before_int_nu=tile_nu_dependence_factor*lineshape_speed_factor_integrated
    
    
    # this should have units of the number of integrand positions
    lineshape_int_over_nu = np.trapz(lineshape_before_int_nu,x=nu_integrand_range)
    
    # this should have length of number of integrand 
    excitation_rate = factor_outside_integral*lineshape_int_over_nu
    
    return(excitation_rate)

def voltage_on_pd(power,responsivity,transimpedance_gain):
    # arguments in W, A/W, V/A
    
    result = power * responsivity * transimpedance_gain
    
    return result