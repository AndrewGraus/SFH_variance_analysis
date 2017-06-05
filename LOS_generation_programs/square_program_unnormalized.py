#Square program unnormalized
#
#This is a program that takes all the simulations and divides them into different
#LOS and writes the ages of the star particles in that los in projection to an
#hdf5 file
#
#This program takes Alex's dwarfs and calculates the SFH in within the "half light radius"
#actually the half stellar mass radius in 3D, and also the galaxy radius defined by Alex as
# 0.1Rvir, both in 3D and projection.
#
#Then the program take the half light radius in projection and divides it into 100 lines of sight
#by laying a grid over the galaxy with 100 bins in it. the side length of this grid is 2*R_half
#such that the grid covers everything in R_half (plus a little extra since it's a grid).
#
#NOTE: this is not the exact program I used to generate the hdf5 programs for most of the analysis
#done already, it's just a more complex version that prints out the corrdiates of the particles
#in addtion to the ages and metallicities (It give the same results). 

import numpy as np
import yt, h5py, re, os
from astropy.cosmology import FlatLambdaCDM
import matplotlib.gridspec as gridspec

def high_low_limit(data, confidence):
    #############################################
    #
    # Input: data - a list of data for which the 
    #        confidence limits will be computed
    #        confidence - the confidence interval
    #        that will be computed so if you
    #        want the 95% confidence interval
    #        that's 0.95
    # 
    # Output: the average, and the upper and lower
    #         limits of the confidence interval
    #
    ##############################################
    n_low = 0.5-confidence/2.0
    n_high = 0.5+confidence/2.0
    n_avg = 0.5
    sorted_data = sorted(data)
    lim_low = sorted_data[int(len(data)*n_low)]
    lim_high = sorted_data[int(len(data)*n_high)]
    lim_avg = sorted_data[int(len(data)*n_avg)]
    #print len(data)*n_high, int(len(data)*n_high)
    return lim_low, lim_high, lim_avg

h = 0.71

'''First Load the data'''
fitts_dwarf_props = np.loadtxt('./NAME_CONVERSION.txt',dtype=object)
Alex_ids = fitts_dwarf_props[:,0]
Alex_X = fitts_dwarf_props[:,2]
Alex_Y = fitts_dwarf_props[:,3]
Alex_Z = fitts_dwarf_props[:,4]
Alex_r_half = fitts_dwarf_props[:,5]
Alex_r_gal = fitts_dwarf_props[:,6]

for particle_file in os.listdir('/home/agraus/data/dwarf_SFH/z_zero_snaps/'):
    file_path = '/home/agraus/data/dwarf_SFH/z_zero_snaps/'+particle_file

    f_split = re.split("[/_]",file_path)
    sim_name = f_split[9]
    print 'staring: '+str(sim_name)

    '''Create hdf5 file'''
    f = h5py.File(file_path)
    f_write = h5py.File('./square_hdf5_coords/data_'+str(sim_name)+'.hdf5','w')
    host_group = f_write.create_group("galaxy_properties")
    data_group = f_write.create_group("LOS_data")
    
    '''create 4 sub groups two for half light radius in projection and radial
    and two for r gal and r gal projection'''

    host_r_half_rad = host_group.create_group('R_half_radial')
    host_r_half_proj = host_group.create_group('R_half_projected')
    host_r_gal_rad = host_group.create_group('R_gal_radial')
    host_r_gal_proj = host_group.create_group('R_gal_projected')

    '''define center and radius'''
    for ii in range(len(Alex_ids)):
        if 'halo'+str(Alex_ids[ii])==sim_name:
            center = [float(Alex_X[ii])*1000.0/h,float(Alex_Y[ii])*1000.0/h,float(Alex_Z[ii])*1000.0/h]
            radius = float(Alex_r_half[ii])/1000.0
            radius_gal = float(Alex_r_gal[ii])/1000.0
    print 'loading stuff'
    star_coords = f['PartType4']['Coordinates'][:]
    star_mass = f['PartType4']['Masses'][:]
    star_age = f['PartType4']['StellarFormationTime'][:] #units are a
    star_metal = f['PartType4']['Metallicity'][:]
    velocities = f['PartType4']['Velocities'][:]

    '''now to start creating the tiles take the effective radius and divide it
    by ten this is the size of each grid  cell, and then I need to make a 
    matrix that is the x and y coordinates of the center of each cell'''

    a_bins = np.linspace(0.0,1.0,100)
    age_hist_tot = []

    '''Now I want to calculate the projected SFH of the host'''
    print 'center: '+str(center)
    print 'radius: '+str(radius)

    print 'calc half mass rad SFH'
    rad_dist = np.sqrt((center[0]-star_coords[:,0])**2.0+(center[1]-star_coords[:,1])**2.0+(center[2]-star_coords[:,2])**2.0)
    print center, radius, radius_gal
    print max(rad_dist), min(rad_dist)
    r_half_rad_select = (rad_dist<radius)
    r_gal_rad_select = (rad_dist<radius_gal)
    
    r_half_rad_age = star_age[r_half_rad_select]
    r_half_rad_metal = star_metal[r_half_rad_select]
    r_half_rad_coords = star_coords[r_half_rad_select]

    '''radial properties of the host within rhalf'''
    host_r_half_rad_age = host_r_half_rad.create_dataset('age',data=r_half_rad_age)
    host_r_half_rad_metal = host_r_half_rad.create_dataset('metallicity',data=r_half_rad_metal)
    host_r_half_rad_coords = host_r_half_rad.create_dataset('coordinates',data=r_half_rad_coords)

    '''radial properties of the host within R_gal'''
    r_gal_rad_age = star_age[r_gal_rad_select]
    r_gal_rad_metal = star_metal[r_gal_rad_select]
    r_gal_rad_coords = star_coords[r_gal_rad_select]

    host_r_gal_rad_age = host_r_gal_rad.create_dataset('age',data=r_gal_rad_age)
    host_r_gal_rad_metal = host_r_gal_rad.create_dataset('metallicity',data=r_gal_rad_metal)
    host_r_gal_rad_coords = host_r_gal_rad.create_dataset('coordinates',data=r_gal_rad_coords)

    print 'calc total SFH'
    '''an overly complicated way of calculating projected distance'''
    vector = [0.0,0.0,1.0]
    n_vec = np.sqrt(vector[0]**2.0+vector[1]**2.0+vector[2]**2.0)
    normalized_vector = [vector[0]/n_vec, vector[1]/n_vec, vector[2]/n_vec]
    u_vector = [normalized_vector[0]*1000.0,normalized_vector[1]*1000.0,normalized_vector[2]*1000.0]
    v = np.asarray(u_vector)
    w = np.asarray([-u_vector[0],-u_vector[1],-u_vector[2]])
    l2 = (v[0]-w[0])**2.0+(v[1]-w[1])**2.0+(v[2]-w[2])**2.0
    #test = np.minimum(1.0,np.dot(pos-v,w-v)/l2)
    sub_pos_new = star_coords-center
    t = np.maximum(0.0,np.minimum(1.0,np.dot(sub_pos_new-v,w-v)/l2))
    new_t = [xx*(w-v) for xx in t]
    projection = v+new_t
    assert len(projection) == len(sub_pos_new)
    ab_dist = np.sqrt((projection[:,0]-sub_pos_new[:,0])**2.0+(projection[:,1]-sub_pos_new[:,1])**2.0+(projection[:,2]-sub_pos_new[:,2])**2.0)
    proj_select = (ab_dist<radius)
    star_age_proj = star_age[proj_select]
    star_metal_proj = star_metal[proj_select]
    star_coords_proj = star_coords[proj_select]

    '''projected quantities within rhalf'''
    host_r_half_proj_age = host_r_half_proj.create_dataset('age',data=star_age_proj)
    host_r_half_proj_metal = host_r_half_proj.create_dataset('metallicity',data=star_metal_proj)
    host_r_half_proj_coords = host_r_half_proj.create_dataset('coordinates',data=star_coords_proj)

    proj_select_gal = (ab_dist<radius_gal)
    star_coords_proj = star_coords[proj_select_gal]
    star_age_proj = star_age[proj_select_gal]
    star_metal_proj = star_metal[proj_select_gal]

    '''projected quantites with r_gal'''
    host_r_gal_proj_age = host_r_gal_proj.create_dataset('age',data=star_age_proj)
    host_r_gal_proj_metal = host_r_gal_proj.create_dataset('metallicity',data=star_metal_proj)
    host_r_gal_proj_coords = host_r_gal_proj.create_dataset('coordinates',data=star_coords_proj)

    print 'number of star particles: '+str(len(star_age_proj))
    age_hist, age_bins = np.histogram(star_age_proj,bins=a_bins,normed=False)
    c_age_hist = np.cumsum(age_hist)
    age_hist_tot.append(c_age_hist)
    '''now here is the algorithm for finding the centers
    This is scaleable'''

    X = np.arange(-0.5,0.6,0.1)
    Y = np.arange(-0.5,0.6,0.1)

    '''need to do -0.5 to 0.5 and then multiply the radius by 2 
    in order to get an even distribution of 100 points'''

    X_gal = X*2.0*radius+center[0]
    Y_gal = Y*2.0*radius+center[1]

    '''now to calculate the radius of each sub circle
    which is just the distance between any two cell
    centers divided by two'''

    small_rad = abs(X_gal[0]-X_gal[1])/2.0
    
    '''calculate the SFH of all 100 sub sections'''
    
    num_los = 0

    Y_gal_rev = Y_gal[::-1]

    print 'calculating 100 SFH'
    for j in range(len(Y_gal)-1):
        for i in range(len(X_gal)-1):
            ii = X_gal[i]
            jj = Y_gal_rev[j]
            new_cen = [ii,jj,center[2]]

            point1 = [ii, jj]
            point2 = [ii+2.0*small_rad, jj]
            point3 = [ii, jj-2.0*small_rad]
            point4 = [ii+2.0*small_rad,jj-2.0*small_rad]

            xpoints = [point1[0],point2[0],point3[0],point4[0]]
            ypoints = [point1[1],point2[1],point3[1],point4[1]]

            num_los = num_los+1

            square_selection = (star_coords[:,0]<max(xpoints))&(star_coords[:,0]>min(xpoints))&(star_coords[:,1]<max(ypoints))&(star_coords[:,1]>min(ypoints))
            square_age = star_age[square_selection]
            square_metal = star_metal[square_selection]
            square_coords = star_coords[square_selection]

            los_data = data_group.create_group('los_'+str(num_los))
            los_age = los_data.create_dataset('age',data=square_age)
            los_metal = los_data.create_dataset('metallicity',data=square_metal)
            los_coords = los_data.create_dataset('coordinates',data=square_coords)
    
