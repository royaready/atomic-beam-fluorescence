# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:15:59 2020
Modified a lot between April 2020 and July 2021
@author: Roy Ready
email: roy.a.ready@gmail.com

This code generates a single fluorescence peak for a given atomic species,
oven geometry, and oven temperature. The default atomic angular distribution 
is the molecular flow limit from Olander. One can readily toggle this to a 
cosine distribution (the limit of an infinite nozzle ratio) or add a new 
angular distribution.

"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ytterbium = Yb, rubidium = Rb, calcium = Ca...
# species = 'Yb'
# species = 'Ca'
species = 'Rb'

# atomic angular distribution
angdist = 'molecularflow'
# angdist = 'cosine'

scantype='test'
# scantype='fullscan'

# main program
def main(species='Rb',angdist='molecularflow',scantype='test'):
    
    start_time = datetime.now()
    print("\n   ******************   \n simulate_ABF starting. Time is %s\n"%start_time)
    
    single_column = 1

    plot_dims,font_size = abf.load_plot_settings(single_column)
    
    # excitation rate integral type. For a directed beam we use the general 
    # formula which includes an integral over the maxwell-boltzmann speed distribution.
    integral = 'maxboltz'

# =============================================================================
# load the atom and apparatus values
# =============================================================================
    
    (nu_o, mass, A_Einstein,tau_o,f_a, vp_coeff,nozzle_radius,
     nozzle_to_intersxn,det_y,det_radius,nozzle_ratio,r_m,oven_temp,
     laser_radius,laser_power) = abf.load_atom_data(species)
    
    if angdist == 'cosine':
        
        nozzle_ratio = np.infty
        print('\n user has selected cosine ang. dist., updating nozzle ratio -> infinity')
    

# =============================================================================
# volume integral settings. Want the smallest max cube side length that
# integrates over all the nonzero stuff and the smallest number of integral
# steps while keeping the computation time reasonable. As a general rule
# 1mm or less microcubes are good, e.g. maxcube_side length = 30 mm and 
# volume_int_steps = 31. Also choose an odd number of steps so that the center step
# is super close to the center of the fluorescence volume.
#
# Provide lengths in meters, frequencies in Hz
# =============================================================================
    
    if scantype == 'fullscan':
        # this is the full scan, takes hours
        [volume_laser_axis_int_steps,max_cube_side_length_laser_axis,
         max_cube_side_length_transverse_axis,
         laser_scan_width,laser_freq_step_size]= [41,16e-3,16e-3,1000*1e6,5e6] # 12 hours for rubidium with collimated beam.
    
    elif scantype == 'test':
        # should run in like a minute
        [volume_laser_axis_int_steps,max_cube_side_length_laser_axis,
         max_cube_side_length_transverse_axis,
         laser_scan_width,laser_freq_step_size]= [7,16e-3,16e-3,1000*1e6,50e6] # 12 hours for rubidium with collimated beam.
    
    
# =============================================================================
#  begin atutomated laser scan settings. This section sets up the laser scan steps
#  and microcube steps. These settings do not need to be modified unless special
#  modifications of the microcube / laser is needed
# =============================================================================
    laser_freq_wrt_res_min = -laser_scan_width/2. # Hz. Laser scan starting point
    laser_freq_wrt_res_max = laser_scan_width/2. # Hz. Laser scan end point
    
    float_steps = np.ceil((laser_freq_wrt_res_max-laser_freq_wrt_res_min)/laser_freq_step_size)
    laser_steps = int(float_steps + np.mod(float_steps,2)+1)
    laser_freq_wrt_res = np.linspace(laser_freq_wrt_res_min,laser_freq_wrt_res_max,laser_steps,axis=0)
    
    
    laser_axis_range = max_cube_side_length_laser_axis
    transverse_axis_range = max_cube_side_length_transverse_axis # m
    
    
    ten_pct_of_scans = int(laser_steps/10.) # for tracking laser scan progress.
    
    x_steps = volume_laser_axis_int_steps
    
    microcube_side_length = laser_axis_range/x_steps
    
    dvol = microcube_side_length**3
    
    x_start = (-1.)*0.499*laser_axis_range
    
    y_start = (-1.)*0.499*transverse_axis_range
    
    x_stop = .501*laser_axis_range
    
    y_stop = .501*transverse_axis_range
    
    
    # make sure odd number of steps so we get close to the origin
    yz_steps = len(np.arange(y_start,y_stop,microcube_side_length))
    yz_steps_definitely_even = yz_steps + np.mod(yz_steps,2)
    
    if yz_steps_definitely_even == yz_steps:
        
        y_start = (-1.)*0.499*(transverse_axis_range+microcube_side_length)
        y_stop = .501*(transverse_axis_range+microcube_side_length)
        yz_steps = len(np.arange(y_start,y_stop,microcube_side_length))
        
    
    z_start = y_start+nozzle_to_intersxn
    z_stop = y_stop + nozzle_to_intersxn
    
    print('steps along  x, y, z: %i, %i, %i\ntotal elements: %i\nmicrocube side length: %.3e meters'%(
        x_steps,yz_steps,yz_steps,x_steps*yz_steps**2,microcube_side_length)
        )
    
# =============================================================================
# end automated settings    
# =============================================================================
    
    def generate_microcube_coordinates(x_start,x_stop,x_steps,y_start,y_stop,y_steps,z_start,z_stop,z_steps):
        
        X,Y,Z = np.mgrid[x_start:x_stop:x_steps*1j,y_start:y_stop:y_steps*1j,z_start:z_stop:z_steps*1j]
        
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        
        return x,y,z
    
    def plot_the_alphas(z,cos_alpha):
        
        alpha = np.arccos(cos_alpha)
        alpha_degrees=alpha*180./np.pi
        alpha_dev_pitwo = (alpha - np.pi/2)*2/np.pi*100
        
        fig = plt.figure(figsize=np.multiply(plot_dims,2))
    
        ax1 = plt.subplot2grid((2,2), (0,0),)
        
        ax1.scatter(z,alpha_dev_pitwo,marker='x',edgecolor='black',linewidth=0.15)
        ax1.tick_params(labelsize=font_size+4)
        plt.ylabel(r'$\alpha - \pi/2$ (\%)',fontsize=font_size+6)
        plt.xlabel(r'$z$\ (m)',fontsize=font_size+6)
        
        ax2 = plt.subplot2grid((2,2), (0,1),)
        
        ax2.scatter(z,cos_alpha,marker='x',edgecolor='black',linewidth=0.15)
        ax2.tick_params(labelsize=font_size+4)
        plt.ylabel(r'$\cos(\alpha)$',fontsize=font_size+6)
        plt.xlabel(r'$z$\ (m)',fontsize=font_size+6)
        
        ax3 = plt.subplot2grid((2,2), (1,1),)
        ax3.scatter(z,alpha_degrees,marker='x',edgecolor='black',linewidth=0.15)
        ax3.tick_params(labelsize=font_size+4)
        plt.ylabel(r'$\alpha$\ (degrees)',fontsize=font_size+6)
        plt.xlabel(r'$z$\ (m)',fontsize=font_size+6)
        
        plt.tight_layout()
        plt.savefig(''.join((species,'-abf-beam-angles.png')))
        plt.close(fig)
        
    def calculate_photon_atom_yield(jm,speed_z,pos,this_excitation_rate_everywhere,solid_angle):
        return np.sum(
            (jm/(speed_z*(pos**2)))*(det_y**2/(abc.a_det))*this_excitation_rate_everywhere*solid_angle*dvol
            )
    
    def calc_pow_detector(speed_z,flux,this_excitation_rate_everywhere,solid_angle):
        return np.sum(dvol*abc.h_Planck*nu_o/speed_z*(
                flux/(4.*np.pi)*this_excitation_rate_everywhere*solid_angle))

           
    print('---\nlaser scan from %.1e MHz to %.1e Mhz wrt nu_o \nstep size %.1e Mhz, %i steps total'%(
        laser_freq_wrt_res_min*1e-6,laser_freq_wrt_res_max*1e-6,laser_freq_step_size*1e-6,laser_steps
        ))
     
    print('\nfluorescence integral volume = %.3e m by %.3e m by %.3e m prism\n microcube side length = %.3e m'%(
        laser_axis_range,transverse_axis_range,transverse_axis_range,microcube_side_length) 
        )
     
    # most probable speed (m/s) 
    mp_speed = abf.most_probable_speed(oven_temp,mass)
     
    # x, y, z arrays of all the microcube positions
    x,y,z = generate_microcube_coordinates(x_start,x_stop,x_steps,y_start,y_stop,yz_steps,z_start,z_stop,yz_steps)
     
    # scalar distance from origin 
    pos = np.sqrt(x**2 + y**2 + z**2)
     
    # index of x=0 position (should be very close to 0)
    xmid = int(np.ceil(len(x)/2))
     
    # angle between beam axis and position we're integrating over
    theta = np.arctan(np.sqrt((x**2+y**2))/(z))
     
    cos_theta = z/pos
     
    # angle between microcube vec and laser vec, defined to be along x
    cos_alpha = (x/pos)
     
    # angular distribution value. I've been using molecular flow limit,
    # Erin will probably want to update with an intermediate flow 
    # distribution.
    if angdist == 'cosine':
    
        jm = cos_theta
         
    else:
         
        jm = abf.jm_dist_elwise(nozzle_ratio,theta)
     
    # z component of most probable speed
    speed_z = cos_theta*mp_speed
     
    # axial position relative to laser axis
    rho = np.sqrt(y**2+(z-nozzle_to_intersxn)**2)
     
    fluorescence_center_index = np.argmin(np.sqrt(x**2 + y**2 + (z-nozzle_to_intersxn)**2))
     
    # this function generates a few subplots of the angle between the atom
    # velocity and laser axis, gives an idea of the effect on Doppler broadening
    plot_the_alphas(z,cos_alpha)
     
    # numerical integration of flat circular detector area
    solid_angle = (
        abf.solid_angle_circular_detector_elwise(det_radius,
        det_y,nozzle_to_intersxn,x,y,z)
        )
     
    # the flux of atoms exiting the oven
    flux = abf.atom_flux_elwise(oven_temp,jm,nozzle_radius,pos,mass,vp_coeff,speed_z)
     
    # This is the start of the laser scan. The first value is outside a loop
    # to initalize the arrays.
     
    print('---\nstarting laser scan\n')
    print('%.2e MHz , time %s'%(laser_freq_wrt_res[0]*1e-6,datetime.now()))
    def the_laser_scan():
    
        excitation_rate_everywhere = np.append([abf.excitation_rate(laser_freq_wrt_res[0],nu_o,A_Einstein,
                        mass,oven_temp,cos_alpha,laser_power,rho,laser_radius,f_a)],
                                [abf.excitation_rate(laser_freq_wrt_res[1],nu_o,A_Einstein,
                        mass,oven_temp,cos_alpha,laser_power,rho,laser_radius,f_a)],axis=0)
    
        # the excitation_rate
        transpose_excitation_rate_everywhere = np.transpose(excitation_rate_everywhere)
         
        # These are the excitation rates at the fluorescence origin, so they should be
        # maximum values
        excitation_rate = transpose_excitation_rate_everywhere[:][fluorescence_center_index]
        
        # eta, the number of photons emitted per atom
        eta = np.sum(
            (jm/(speed_z*(pos**2)))*(det_y**2/(abc.a_det))*excitation_rate_everywhere*solid_angle*dvol,
            axis=1
            )
        
        # the power incident on the photodetector               
        pow_detector = np.sum(dvol*abc.h_Planck*nu_o/speed_z*(
                    flux/(4.*np.pi)*excitation_rate_everywhere*solid_angle),
                    axis=1)
        
        scan_count = 1
        
        # The rest of the laser frequency steps are handled in a FOR loop.
        for this_laser_freq in laser_freq_wrt_res[2:]:
            
            scan_count = scan_count + 1
            
            if np.mod(scan_count,ten_pct_of_scans) == 0:
                
                print('%.2e MHz , time %s'%(this_laser_freq*1e-6,datetime.now()))
         
            # the excitation rate, also equal to fluorescence rate in weak pumping limit
            this_excitation_rate_everywhere = abf.excitation_rate(this_laser_freq,
                    nu_o,A_Einstein,mass,oven_temp,cos_alpha,laser_power,rho,laser_radius,f_a)
            
            # for the datafile, we want to see the peak excitation rate, i.e. center of fluorescence region
            excitation_rate = np.append(excitation_rate,
                        this_excitation_rate_everywhere[fluorescence_center_index])
             
            # photons per atom eta
            eta = np.append(eta,[calculate_photon_atom_yield(jm,speed_z,pos,
                        this_excitation_rate_everywhere,
                        solid_angle)],axis=0)
            
            # power incident on the detector
            pow_detector = np.append(pow_detector,
                [calc_pow_detector(speed_z,flux,this_excitation_rate_everywhere,
                    solid_angle)],axis=0)
         
        # END laser scan
        
        return excitation_rate,eta,pow_detector

    excitation_rate,eta,pow_detector, = the_laser_scan()
     
    # voltage on the photodetector
    pd_voltage = abf.voltage_on_pd(pow_detector,r_m,abc.g)
    
    # number density of the oven
    o_num_density = abf.oven_number_density(vp_coeff,oven_temp)
    
    # the rate of atoms exiting the oven
    atom_exit_rate = abf.oven_atom_exit_rate(o_num_density,
                                            speed_z,nozzle_radius)
    max_atom_exit_rate = np.max(atom_exit_rate)
     
    origin_solid_angle = solid_angle[xmid]
     
     
# =============================================================================
# Calculate how long program took.
# =============================================================================
                 
    end_time = datetime.now()
    print("\n Finished calculations. Time is %s\n   ******************  \n"%end_time)
    
    total_time = end_time - start_time
    
    total_time_days = total_time.days
    total_time_minutes = float(total_time.seconds/60.)    
    total_time_seconds = float(total_time.seconds)
    print("\n Calculations took %i days and %.1f minutes (%.1f seconds) to run.\n "%(
        total_time_days,total_time_minutes,total_time_seconds))
    
    save_timestamp = datetime.now().isoformat(timespec='seconds',sep='-').replace(":","")
    
    # These are some useful plots that can be used to check that the 
    # calculated excitation rate makes sense.
    
    fig, ax = plt.subplots(figsize=plot_dims)
    plt.scatter(laser_freq_wrt_res*1e-6,excitation_rate,marker='o',facecolors='none',
                edgecolor="red")
    plt.title('%s abf sim excitation rate, $2r/L=$ %.3f'%(species,nozzle_ratio))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylabel(r'$R$ (s$^{-1}$)')
    plt.xlabel(r'laser freq $- \nu_0$ (MHz)')
    plt.tight_layout()
    plt.savefig(''.join((save_timestamp,'-',species,'-abfsim-R-exc.pdf')))
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=plot_dims)
    plt.scatter(laser_freq_wrt_res*1e-6,pow_detector,marker='o',facecolors='none',
                edgecolor="red")
    plt.title('%s abf sim photodetector power, $2r/L=$ %.3f'%(species,nozzle_ratio))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylabel(r'$P_{\gamma}$ (W)')
    plt.xlabel(r'laser freq $- \nu_0$ (MHz)')
    plt.tight_layout()
    plt.savefig(''.join((save_timestamp,'-',species,'-abfsim-power.pdf')))
    plt.close(fig)
     
    fig, ax = plt.subplots(figsize=plot_dims)
    plt.scatter(laser_freq_wrt_res*1e-6,eta,marker='o',facecolors='none',
                edgecolor="red")
    plt.title('%s abf sim photon-atom yield, $2r/L=$ %.3f'%(species,nozzle_ratio))
    plt.ylabel(r'$\eta$ (photons / atom)')
    plt.xlabel(r'laser freq $- \nu_0$ (MHz)')
    plt.tight_layout()
    plt.savefig(''.join((save_timestamp,'-',species,'-abfsim-eta-yield.pdf')))
    plt.close(fig)
     
    # Save tab-delimited arrays of important calculations at each laser frequency
    
    text_filename = ''.join((save_timestamp,'-',species,"-abfsim-",integral,".txt"))
    text_file = open(text_filename, "w")
     
    text_file.write('analysis ran at: \t %s\n'%datetime.now())
    text_file.write('species: \t %s\n'%species)
    text_file.write(
            "laser freq (Hz) \t APD (V) \t power (W) \t eta \t atom  rate (1/s) \t Omega (rad) \t R_excite (s^-1) \t laser power (mW) \t laser radius (mm) \t oven temp (C) \t nozzle\n")
    
    for i in range(len(laser_freq_wrt_res)):
        
        text_file.write("%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e\n" % (
            laser_freq_wrt_res[i],pd_voltage[i],pow_detector[i],eta[i],
            max_atom_exit_rate,origin_solid_angle,excitation_rate[i],
            laser_power*1e3,laser_radius*1e3,oven_temp,nozzle_ratio)
            )
     
    print('---\nsaving file %s'%text_filename)
    text_file.close()  
    
if __name__ == "__main__":
    import sys
    import abf_constants as abc
    import abf_eqns as abf
#    main(sys.argv[1:])

    main(species,angdist,scantype)
        