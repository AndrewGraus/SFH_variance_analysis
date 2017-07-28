#The idea behind this is to make a package with all the modules I need
#I think to start I only really need two
#
#1) Port the program that calculates the SFH variance as a function of coverage fraction
#2) Port the program that bootstraps that variance calculation

def find_t50(input_los):
    import astroML.resample
    import simple_tools
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)
    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]
    t_halfs_list = []
    if input_los.ndim==1:
        selected_los = input_los
        age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
        c_age_hist = np.cumsum(age_hist)
        constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
        con_t_half_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.5]
        return min(con_t_half_list)
    else:
        for ii in range(len(input_los[:,0])):
            selected_los = input_los[ii]
            age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
            c_age_hist = np.cumsum(age_hist)
            constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
            con_t_half_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.5]
            t_halfs_list.append(min(con_t_half_list))
        return t_halfs_list

def find_t90(input_los):
    import astroML.resample
    import simple_tools
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)
    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]
    t_90_list = []
    if input_los.ndim==1:
        selected_los = input_los
        age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
        c_age_hist = np.cumsum(age_hist)
        constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
        con_t_90_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.9]
        return min(con_t_90_list)
    else:
        for ii in range(len(input_los[:,0])):
            selected_los = input_los[ii]
            age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
            c_age_hist = np.cumsum(age_hist)
            constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
            con_t_90_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.9]
            t_90_list.append(min(con_t_90_list))
        return t_90_list
    
def find_tq(input_los):
    import astroML.resample
    import simple_tools
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)
    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]
    t_q_list = []
    if input_los.ndim==1:
        selected_los = input_los
        age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
        c_age_hist = np.cumsum(age_hist)
        constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
        con_t_q_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=1.0]
        return min(con_t_q_list)
    else:
        for ii in range(len(input_los[:,0])):
            selected_los = input_los[ii]
            age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
            c_age_hist = np.cumsum(age_hist)
            constructed_sfh_norm = c_age_hist/float(max(c_age_hist))
            con_t_q_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=1.0]
            t_q_list.append(min(con_t_q_list))
        return t_q_list

def SFH_variance(hdf5_file,coverage_bins=[1,5,10,20,30,40,50,60,70,80,90,99],coverage_iterations=1000):
    import numpy as np
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    from random import sample
    from scipy.stats.mstats import chisquare
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    #SFH_variance
    #
    #This program takes a hdf5 file generated dividing the stellar distribution into different LOS
    #and generates the variance as a function of coverage fraction
    #
    #INPUTS:
    # hdf5_file - the file to do the analysis on
    # coverage_bins - the percentages to do the calculation on

    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]

    coverage_list = coverage_bins

    SFH_data = h5py.File(hdf5_file)
    
    sfh_min, sfh_max, sfh_mean = [],[],[]
    sfh_low, sfh_high = [],[]
    
    rms_err_min, rms_err_max, rms_err_mean = [],[],[]
    
    con_t_half_min, con_t_half_max, con_t_half_mean, con_t_half_low, con_t_half_high = [],[],[],[],[]
    con_t_90_min, con_t_90_max, con_t_90_mean, con_t_90_low, con_t_90_high = [],[],[],[],[]
    con_t_q_min, con_t_q_max, con_t_q_mean, con_t_q_low, con_t_q_high = [],[],[],[],[]
    
    #For each fraction in the coverage list
    for ii in coverage_list:
        con_t_half,con_t_90,con_t_q=[],[],[]
        #do this many times 
        for jj in range(coverage_iterations):
            #grab a number of los corresponding to the coverage fraction
            #each los is ~1% (its actually 1.27% because we use a square not 
            #a circle of radius R_half)
            #of the galaxy thus the coverage fraction is the
            #percent coverage of the half light radius
            randint_list = sample(xrange(1,100), ii)
            constructed_sfh = [0.0 for xx in range(len(T_bins_fix))]
            constructed_sfh = np.asarray(constructed_sfh)
            for kk in randint_list:
                '''Now I need to grab the correct los'''
                selected_los = SFH_data['LOS_data']['los_'+str(kk)]['age'][:]
                age_hist, age_bin = np.histogram(selected_los,bins=a_bins,normed=False)
                c_age_hist = np.cumsum(age_hist)
                #add cumulative histogram to all the others
                constructed_sfh = constructed_sfh+np.asarray(c_age_hist)
            if max(constructed_sfh) == 0.0:
                #if its for some reason zero just set the whole thing to zero (this is really only a problem at 1%
                constructed_sfh_norm = [0.0 for xx in range(len(T_bins_fix))]
            else:
                #normalize it
                constructed_sfh_norm = constructed_sfh/float(max(constructed_sfh))
            #calculate t50 t90 and tq
            con_t_half_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.5]
            con_t_90_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.9]
            con_t_q_list = [T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=1.0]
            if len(con_t_half_list)==0:
                con_t_half.append(13.7)
            else:
                con_t_half.append(min(con_t_half_list))
            if len(con_t_90_list)==0:
                con_t_90.append(13.7)
            else:
                con_t_90.append(min(con_t_90_list))
            if len(con_t_q_list)==0:
                con_t_q.append(13.7)
            else:
                con_t_q.append(min(con_t_q_list))
        con_t_half_min.append(min(con_t_half))
        con_t_half_max.append(max(con_t_half))
        con_t_half_mean.append(np.mean(con_t_half))
        half_low, half_high, half_med = high_low_limit(con_t_half,0.68)
        con_t_half_low.append(half_low)
        con_t_half_high.append(half_high)
        
        con_t_q_min.append(min(con_t_q))
        con_t_q_max.append(max(con_t_q))
        con_t_q_mean.append(np.mean(con_t_q))
        q_low, q_high, q_med = high_low_limit(con_t_q,0.68)
        con_t_q_low.append(q_low)
        con_t_q_high.append(q_high)
        
        con_t_90_min.append(min(con_t_90))
        con_t_90_max.append(max(con_t_90))
        con_t_90_mean.append(np.mean(con_t_90))
        nine_low, nine_high, nine_med = high_low_limit(con_t_90,0.68)
        con_t_90_low.append(nine_low)
        con_t_90_high.append(nine_high)

    #for now the mean is the mean of the distribution which is probably not a great idea, it should be the mean of the
    #actual SFH
    
    con_t_half_err_low = [(con_t_half_low[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_low))]
    con_t_half_err_high = [(con_t_half_high[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_high))]

    con_t_q_err_low = [(con_t_q_low[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_low))]
    con_t_q_err_high = [(con_t_q_high[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_high))]
    
    con_t_90_err_low = [(con_t_90_low[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_low))]
    con_t_90_err_high = [(con_t_90_high[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_high))]

    con_t_half_err_min = [(con_t_half_min[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_min))]
    con_t_half_err_max = [(con_t_half_max[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_max))]

    con_t_q_err_min = [(con_t_q_min[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_min))]
    con_t_q_err_max = [(con_t_q_max[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_max))]

    con_t_90_err_min = [(con_t_90_min[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_min))]
    con_t_90_err_max = [(con_t_90_max[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_max))]

    con_t_matrix = np.zeros((len(coverage_list),12))
    con_t_matrix[:,0] = con_t_half_err_low
    con_t_matrix[:,1] = con_t_half_err_high
    con_t_matrix[:,2] = con_t_half_err_min
    con_t_matrix[:,3] = con_t_half_err_max

    con_t_matrix[:,4] = con_t_90_err_low
    con_t_matrix[:,5] = con_t_90_err_high
    con_t_matrix[:,6] = con_t_90_err_min
    con_t_matrix[:,7] = con_t_90_err_max

    con_t_matrix[:,8] = con_t_q_err_low
    con_t_matrix[:,9] = con_t_q_err_high
    con_t_matrix[:,10] = con_t_q_err_min
    con_t_matrix[:,11] = con_t_q_err_max

    return con_t_matrix


def SFH_variance_bootstrap(hdf5_file,coverage_bins=[1,5,10,20,30,40,50,60,70,80,90,99],coverage_iterations=1000,bs_iterations=1000):
    '''fiducial scatter from SFH variation'''
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    import numpy as np
    from random import sample
    from scipy.stats.mstats import chisquare
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]

    coverage_list = coverage_bins

    tot_half_low_list, tot_half_high_list = [],[]
    tot_90_low_list, tot_90_high_list = [],[]
    tot_q_low_list, tot_q_high_list = [],[]

    SFH_data = h5py.File(hdf5_file)

    sfh_min, sfh_max, sfh_mean = [],[],[]
    sfh_low, sfh_high = [],[]
    
    rms_err_min, rms_err_max, rms_err_mean = [],[],[]

    con_t_half_min, con_t_half_max, con_t_half_mean, con_t_half_low, con_t_half_high = [],[],[],[],[]
    con_t_90_min, con_t_90_max, con_t_90_mean, con_t_90_low, con_t_90_high = [],[],[],[],[]
    con_t_q_min, con_t_q_max, con_t_q_mean, con_t_q_low, con_t_q_high = [],[],[],[],[]

    bs_t_half_min, bs_t_half_max, bs_t_half_mean, bs_t_half_low, bs_t_half_high = [],[],[],[],[]
    bs_t_90_min, bs_t_90_max, bs_t_90_mean, bs_t_90_low, bs_t_90_high = [],[],[],[],[]
    bs_t_q_min, bs_t_q_max, bs_t_q_mean, bs_t_q_low, bs_t_q_high = [],[],[],[],[]

    for ii in coverage_list:
        con_t_half,con_t_90,con_t_q=[],[],[]
        bs_t_half, bs_t_90, bs_t_q=[],[],[]
        for jj in range(coverage_iterations):
            randint_list = sample(xrange(1,100), ii)
            ages_list = []
            for kk in randint_list:
                '''Now I need to grab the correct los'''
                target_los = SFH_data['LOS_data']['los_'+str(kk)]['age'][:]
                [ages_list.append(xx) for xx in target_los]
        
            ages_list = np.asarray(ages_list)
            actual_t50 = find_t50(ages_list)
            actual_t90 = find_t90(ages_list)
            actual_tq = find_tq(ages_list)
    
            bs_t50s = astroML.resample.bootstrap(ages_list, bs_iterations, find_t50)
            bs_t90s = astroML.resample.bootstrap(ages_list, bs_iterations, find_t90)
            bs_tqs = astroML.resample.bootstrap(ages_list, bs_iterations, find_tq)
        
            con_t_half.append(actual_t50)
            con_t_90.append(actual_t90)
            con_t_q.append(actual_tq)
        
            [bs_t_half.append(xx) for xx in bs_t50s]
            [bs_t_90.append(xx) for xx in bs_t90s]
            [bs_t_q.append(xx) for xx in bs_tqs]
    
        #actual distribution
        con_t_half_min.append(min(con_t_half))
        con_t_half_max.append(max(con_t_half))
        con_t_half_mean.append(np.mean(con_t_half))
        half_low, half_high, half_med = high_low_limit(con_t_half,0.68)
        con_t_half_low.append(half_low)
        con_t_half_high.append(half_high)

        con_t_q_min.append(min(con_t_q))
        con_t_q_max.append(max(con_t_q))
        con_t_q_mean.append(np.mean(con_t_q))
        q_low, q_high, q_med = high_low_limit(con_t_q,0.68)
        con_t_q_low.append(q_low)
        con_t_q_high.append(q_high)

        con_t_90_min.append(min(con_t_90))
        con_t_90_max.append(max(con_t_90))
        con_t_90_mean.append(np.mean(con_t_90))
        nine_low, nine_high, nine_med = high_low_limit(con_t_90,0.68)
        con_t_90_low.append(nine_low)
        con_t_90_high.append(nine_high)
    
        #boostrapped distribution
    
        bs_t_half_min.append(min(bs_t_half))
        bs_t_half_max.append(max(bs_t_half))
        bs_t_half_mean.append(np.mean(bs_t_half))
        half_low, half_high, half_med = high_low_limit(bs_t_half,0.68)
        bs_t_half_low.append(half_low)
        bs_t_half_high.append(half_high)

        bs_t_q_min.append(min(bs_t_q))
        bs_t_q_max.append(max(bs_t_q))
        bs_t_q_mean.append(np.mean(bs_t_q))
        q_low, q_high, q_med = high_low_limit(bs_t_q,0.68)
        bs_t_q_low.append(q_low)
        bs_t_q_high.append(q_high)

        bs_t_90_min.append(min(bs_t_90))
        bs_t_90_max.append(max(bs_t_90))
        bs_t_90_mean.append(np.mean(bs_t_90))
        nine_low, nine_high, nine_med = high_low_limit(bs_t_90,0.68)
        bs_t_90_low.append(nine_low)
        bs_t_90_high.append(nine_high)

    #actual error distribution

    con_t_half_err_low = [(con_t_half_low[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_low))]
    con_t_half_err_high = [(con_t_half_high[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_high))]
        
    con_t_q_err_low = [(con_t_q_low[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_low))]
    con_t_q_err_high = [(con_t_q_high[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_high))]
    
    con_t_90_err_low = [(con_t_90_low[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_low))]
    con_t_90_err_high = [(con_t_90_high[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_high))]

    con_t_half_err_min = [(con_t_half_min[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_min))]
    con_t_half_err_max = [(con_t_half_max[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_max))]

    con_t_q_err_min = [(con_t_q_min[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_min))]
    con_t_q_err_max = [(con_t_q_max[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_max))]

    con_t_90_err_min = [(con_t_90_min[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_min))]
    con_t_90_err_max = [(con_t_90_max[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_max))]

    #bootstrapped errors

    bs_t_half_err_low = [(bs_t_half_low[xx]-bs_t_half_mean[xx])/bs_t_half_mean[xx] for xx in range(len(bs_t_half_low))]
    bs_t_half_err_high = [(bs_t_half_high[xx]-bs_t_half_mean[xx])/bs_t_half_mean[xx] for xx in range(len(bs_t_half_high))]
    
    bs_t_q_err_low = [(bs_t_q_low[xx]-bs_t_q_mean[xx])/bs_t_q_mean[xx] for xx in range(len(bs_t_q_low))]
    bs_t_q_err_high = [(bs_t_q_high[xx]-bs_t_q_mean[xx])/bs_t_q_mean[xx] for xx in range(len(bs_t_q_high))]

    bs_t_90_err_low = [(bs_t_90_low[xx]-bs_t_90_mean[xx])/bs_t_90_mean[xx] for xx in range(len(bs_t_90_low))]
    bs_t_90_err_high = [(bs_t_90_high[xx]-bs_t_90_mean[xx])/bs_t_90_mean[xx] for xx in range(len(bs_t_90_high))]
    
    bs_t_half_err_min = [(bs_t_half_min[xx]-bs_t_half_mean[xx])/bs_t_half_mean[xx] for xx in range(len(bs_t_half_min))]
    bs_t_half_err_max = [(bs_t_half_max[xx]-bs_t_half_mean[xx])/bs_t_half_mean[xx] for xx in range(len(bs_t_half_max))]

    bs_t_q_err_min = [(bs_t_q_min[xx]-bs_t_q_mean[xx])/bs_t_q_mean[xx] for xx in range(len(bs_t_q_min))]
    bs_t_q_err_max = [(bs_t_q_max[xx]-bs_t_q_mean[xx])/bs_t_q_mean[xx] for xx in range(len(bs_t_q_max))]

    bs_t_90_err_min = [(bs_t_90_min[xx]-bs_t_90_mean[xx])/bs_t_90_mean[xx] for xx in range(len(bs_t_90_min))]
    bs_t_90_err_max = [(bs_t_90_max[xx]-bs_t_90_mean[xx])/bs_t_90_mean[xx] for xx in range(len(bs_t_90_max))]

    con_t_matrix = np.zeros((len(coverage_list),24))
    con_t_matrix[:,0] = con_t_half_err_low
    con_t_matrix[:,1] = bs_t_half_err_low
    con_t_matrix[:,2] = con_t_half_err_high
    con_t_matrix[:,3] = bs_t_half_err_high
    con_t_matrix[:,4] = con_t_half_err_min
    con_t_matrix[:,5] = bs_t_half_err_min
    con_t_matrix[:,6] = con_t_half_err_max
    con_t_matrix[:,7] = bs_t_half_err_max

    con_t_matrix[:,8] = con_t_90_err_low
    con_t_matrix[:,9] = bs_t_90_err_low
    con_t_matrix[:,10] = con_t_90_err_high
    con_t_matrix[:,11] = bs_t_90_err_high
    con_t_matrix[:,12] = con_t_90_err_min
    con_t_matrix[:,13] = bs_t_90_err_min
    con_t_matrix[:,14] = con_t_90_err_max
    con_t_matrix[:,15] = bs_t_90_err_max

    con_t_matrix[:,16] = con_t_q_err_low
    con_t_matrix[:,17] = bs_t_q_err_low
    con_t_matrix[:,18] = con_t_q_err_high
    con_t_matrix[:,19] = bs_t_q_err_high
    con_t_matrix[:,20] = con_t_q_err_min
    con_t_matrix[:,21] = bs_t_q_err_min
    con_t_matrix[:,22] = con_t_q_err_max
    con_t_matrix[:,23] = bs_t_q_err_max

    return con_t_matrix

def scatter_versus_random(hdf5_file,coverage_list=[1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,99],coverage_trials=1000,random_trials=1000):
    #The idea behind this module is to take a certain sample of "fields" and then count up the number of particles
    #within these "fields" measuring taking the same number of particles randomly from all the los and then measure
    #the parameters for the mock observations and the random sample.  The idea is that if the difference in the 
    #parameters is not larger than the one from the random sample you really can't tell the difference from
    #pure randomness
    #
    #Inputs:
    #
    #hdf5_file - hdf5 file containing the stellar distribution divided up into different lines of sight
    #
    #coverage_list - list of percent coverage values just the number of fields to randomly select
    #
    #coverage_trials - the number of trials to do for each value of the coverage list.  We want to do this
    #                  in order to build up number statistics. Theoretically we could do this for every
    #                  permutation of fields. which would be an N choose k problem this would be 100 for 1 
    #                  field but 10^29 (!!) for 100 choose 50. Maybe do this in a way that's tied to the
    #                  coverage fraction, in some way
    #
    #random_trails - number of trials to compare each field to in the random sample (how do I set this?)
    #
    #
    #returns:
    #
    # N_particles - list containing the number of particles
    # 
    # coverage_fraction - list containing the coverage fraction of all the trials (this will probably be discrete to begin with)
    # 
    # t_<n>_error - the error for the number parameter 50 for t50, 90 for t90 and 100 for t100
    #
    # t_<n>_error_rand_<lim> - the error for the random sample. This has to be done multiple times, so the final results are given
    #                          in terms of various limits.  For now 1-sigma low and high, min and max.
    #
    #

    '''fiducial scatter from SFH variation'''
    from random import sample
    from scipy.stats.mstats import chisquare
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    import numpy as np
    from astropy.cosmology import FlatLambdaCDM

    SFH_data = h5py.File(hdf5_file)

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    a_bins = np.linspace(0.0,1.0,1000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]

    #go ahead and calculate the t parameters for the total galaxy distribution in the fields

    all_star_ages = []

    for los_key in SFH_data['LOS_data'].keys():
        '''dump all the particles into one list'''
        target_los = SFH_data['LOS_data'][los_key]['age'][:]
        [all_star_ages.append(xx) for xx in target_los]

    total_age_hist, total_age_bin = np.histogram(all_star_ages,bins=a_bins,normed=False)
    c_age_hist = np.cumsum(total_age_hist)
    total_sfh_norm = c_age_hist/float(max(c_age_hist))
    total_t_half = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.5])
    total_t_90 = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.9])
    total_t_q = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.99])

    tot_half_low_list, tot_half_high_list = [],[]
    tot_90_low_list, tot_90_high_list = [],[]
    tot_q_low_list, tot_q_high_list = [],[]

    tot_t_50s, tot_t_90s, tot_t_qs = [],[],[]
    N_tot_list = []
    coverage_tot_list = []
    
    for ii in coverage_list:
        for jj in range(coverage_trials):
            randint_list = sample(xrange(1,100), ii)
            N_tot = 0
            constructed_sfh = [0.0 for xx in range(len(T_bins_fix))]
            constructed_sfh = np.asarray(constructed_sfh)
            for kk in randint_list:
                '''Now I need to grab the correct los'''
                target_los = SFH_data['LOS_data']['los_'+str(kk)]['age'][:]
                N_tot = N_tot + len(target_los)
                age_hist, age_bin = np.histogram(target_los,bins=a_bins,normed=False)
                c_age_hist = np.cumsum(age_hist) 
                constructed_sfh = constructed_sfh+np.asarray(c_age_hist)
            N_tot_list.append(N_tot)
            coverage_tot_list.append(ii)
            constructed_sfh_norm = constructed_sfh/float(max(constructed_sfh))
            con_t_half_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.5])
            con_t_90_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.9])
            con_t_q_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=1.0])
            tot_t_50s.append(con_t_half_list)
            tot_t_90s.append(con_t_90_list)
            tot_t_qs.append(con_t_q_list)
        
    #now for every N in N_tot_list calculate the average error for that number of particles

    tot_random_t_half_min, tot_random_t_90_min, tot_random_t_q_min = [], [], []
    tot_random_t_half_max, tot_random_t_90_max, tot_random_t_q_max = [], [], []
    tot_random_t_half_med, tot_random_t_90_med, tot_random_t_q_med = [], [], []

    tot_random_t_half_low, tot_random_t_90_low, tot_random_t_q_low = [], [], []
    tot_random_t_half_high, tot_random_t_90_high, tot_random_t_q_high = [], [], []

    #do the random trials for all the particle numbers in N_tot_list

    for N_part in N_tot_list:
        random_t_half, random_t_90, random_t_q = [], [], []
        for kk in range(random_trials):
            random_SFH = sample(all_star_ages, N_part)
            age_hist, age_bin = np.histogram(random_SFH,bins=a_bins,normed=False)
            c_age_hist = np.cumsum(age_hist)
            c_sfh_norm = c_age_hist/float(max(c_age_hist))
            con_t_half_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=0.5]
            con_t_90_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=0.9]
            con_t_q_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=1.0]

        
            random_t_half.append(min(con_t_half_list))
            random_t_90.append(min(con_t_90_list))
            random_t_q.append(min(con_t_q_list))
    
        low, high, med = high_low_limit(random_t_half,0.68)
        tot_random_t_half_low.append(low)
        tot_random_t_half_high.append(high)
        tot_random_t_half_med.append(med)
        tot_random_t_half_min.append(min(random_t_half))
        tot_random_t_half_max.append(max(random_t_half))
    
        low, high, med = high_low_limit(random_t_90,0.68)
        tot_random_t_90_low.append(low)
        tot_random_t_90_high.append(high)
        tot_random_t_90_med.append(med)
        tot_random_t_90_min.append(min(random_t_90))
        tot_random_t_90_max.append(max(random_t_90))
    
        low, high, med = high_low_limit(random_t_q,0.68)
        tot_random_t_q_low.append(low)
        tot_random_t_q_high.append(high)
        tot_random_t_q_med.append(med)
        tot_random_t_q_min.append(min(random_t_q))
        tot_random_t_q_max.append(max(random_t_q))

    results_matrix = np.zeros(len(N_tot_list),20)
    results_matrix[:,0] = N_tot_list
    results_matrix[:,1] = coverage_tot_list
    results_matrix[:,2] = total_t_half
    results_matrix[:,3] = total_t_90
    results_matrix[:,4] = total_t_q
    results_matrix[:,5] = tot_t_50s
    results_matrix[:,6] = tot_t_90s
    results_matrix[:,7] = tot_t_qs
    results_matrix[:,8] = tot_random_t_half_min
    results_matrix[:,9] = tot_random_t_half_low
    results_matrix[:,10] = tot_random_t_half_high
    results_matrix[:,11] = tot_random_t_half_max
    results_matrix[:,12] = tot_random_t_90_min
    results_matrix[:,13] = tot_random_t_90_low
    results_matrix[:,14] = tot_random_t_90_high
    results_matrix[:,15] = tot_random_t_90_max
    results_matrix[:,16] = tot_random_t_q_min
    results_matrix[:,17] = tot_random_t_q_low
    results_matrix[:,18] = tot_random_t_q_high
    results_matrix[:,19] = tot_random_t_q_max

    return results_matrix

def coverage_to_N_converter(hdf5_file,coverage_list,N_trials=10000):
    from scipy.stats.mstats import chisquare
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    import numpy as np
    from random import sample
    from scipy.stats.mstats import chisquare
    from astropy.cosmology import FlatLambdaCDM
    '''Given a hdf5 file formated into fields that cover ~1% of the galaxy
    what is the variance in particle number over the whole galaxy'''

    '''Input:
    
    hdf5_file - the file formatted for SFH analysis
    coverage_list - a list of coverage fractions (number of fields to use
                    which is just a number from 1 to 100
    N_trials - the number of times to do this in order to get a decent idea
               of the variance
    
    returns:

    N_mean_list, N_max_list, N_min_list - lists giving the mean and distribution
    of number of particles as a function of coverage fraction'''

    SFH_data = h5py.File(hdf5_file)

    N_tot_mean, N_tot_min, N_tot_max = [],[],[]

    for ii in coverage_list:
        N_tot_list = []
        for jj in range(N_trials):
            randint_list = sample(xrange(1,100), ii)
            N_tot = 0
            for kk in randint_list:
                '''Now I need to grab the correct los'''
                target_los = SFH_data['LOS_data']['los_'+str(kk)]['age'][:]
                N_tot = N_tot + len(target_los)
            N_tot_list.append(N_tot)
        
        N_tot_mean.append(np.mean(N_tot_list))
        N_tot_min.append(min(N_tot_list))
        N_tot_max.append(max(N_tot_list))

    return N_tot_mean, N_tot_min, N_tot_max

def calculate_random_number_variance(hdf5_file,coverage_list,N_trials):
    '''What is the error for a random selection of particles
    basically take in a coverage fraction take the mean number of particles
    for that coverage fraction, randomly select that number of particles from
    the total galaxy (r_half in projection) and then calculate the variance
    in parameters for that.'''

    '''Input:
    hdf5_file - the hdf5 file formated for this sort of thing
    coverage_list - the number of fields to randomly select
    N_trails - total number of times to do this (should this be Npart dependent?

    returns:
    t_50_matrix - a matrix giving the properties for the t_50 parameter
                  columns corresponding to coverage fraction, number of
                  particles, t_50_mean, t_50_low and high (68% confidence),
                  and t_50 min max (the min and max of all trials)

    t_90_matrix - same as above but for t_90
    
    t_q_matrix - same but for t_100'''

    from random import sample
    from scipy.stats.mstats import chisquare
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    import numpy as np
    from random import sample
    from scipy.stats.mstats import chisquare
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    a_bins = np.linspace(0.0,1.0,10000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]

    SFH_data = h5py.File(hdf5_file)

    all_star_ages = []

    for los_key in SFH_data['LOS_data'].keys():
        '''dump all the particles into one list'''
        target_los = SFH_data['LOS_data'][los_key]['age'][:]
        [all_star_ages.append(xx) for xx in target_los]

    total_age_hist, total_age_bin = np.histogram(all_star_ages,bins=a_bins,normed=False)
    c_age_hist = np.cumsum(total_age_hist) 
    total_sfh_norm = c_age_hist/float(max(c_age_hist))
    total_t_half = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.5])
    total_t_90 = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.9])
    total_t_q = np.min([T_bins_fix[xx] for xx in range(len(total_sfh_norm)) if total_sfh_norm[xx]>=0.99])

    N_tot_mean, N_tot_min, N_tot_max = coverage_to_N_converter(hdf5_file,coverage_list,N_trials=10000)

    tot_t_50s, tot_t_90s, tot_t_qs = [],[],[]
    N_tot_list = []

    tot_random_t_half_min, tot_random_t_90_min, tot_random_t_q_min = [], [], []
    tot_random_t_half_max, tot_random_t_90_max, tot_random_t_q_max = [], [], []
    tot_random_t_half_med, tot_random_t_90_med, tot_random_t_q_med = [], [], []
    
    tot_random_t_half_low, tot_random_t_90_low, tot_random_t_q_low = [], [], []
    tot_random_t_half_high, tot_random_t_90_high, tot_random_t_q_high = [], [], []

    for N_part in N_tot_mean:
        random_t_half, random_t_90, random_t_q = [], [], []
        for kk in range(N_trials):
            random_SFH = sample(all_star_ages, int(N_part))
            age_hist, age_bin = np.histogram(random_SFH,bins=a_bins,normed=False)
            c_age_hist = np.cumsum(age_hist)
            c_sfh_norm = c_age_hist/float(max(c_age_hist))
            con_t_half_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=0.5]
            con_t_90_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=0.9]
            con_t_q_list = [T_bins_fix[xx] for xx in range(len(c_age_hist)) if c_sfh_norm[xx]>=1.0]
        
            random_t_half.append(min(con_t_half_list))
            random_t_90.append(min(con_t_90_list))
            random_t_q.append(min(con_t_q_list))
    
        low, high, med = high_low_limit(random_t_half,0.68)
        tot_random_t_half_low.append(low)
        tot_random_t_half_high.append(high)
        tot_random_t_half_med.append(med)
        tot_random_t_half_min.append(min(random_t_half))
        tot_random_t_half_max.append(max(random_t_half))
    
        low, high, med = high_low_limit(random_t_90,0.68)
        tot_random_t_90_low.append(low)
        tot_random_t_90_high.append(high)
        tot_random_t_90_med.append(med)
        tot_random_t_90_min.append(min(random_t_90))
        tot_random_t_90_max.append(max(random_t_90))
    
        low, high, med = high_low_limit(random_t_q,0.68)
        tot_random_t_q_low.append(low)
        tot_random_t_q_high.append(high)
        tot_random_t_q_med.append(med)
        tot_random_t_q_min.append(min(random_t_q))
        tot_random_t_q_max.append(max(random_t_q))

    t_half_matrix = np.zeros(len(tot_random_t_half_low),5)
    t_half_matrix[:,0] = tot_random_t_half_med
    t_half_matrix[:,1] = tot_random_t_half_low
    t_half_matrix[:,2] = tot_random_t_half_high
    t_half_matrix[:,3] = tot_random_t_half_min
    t_half_matrix[:,4] = tot_random_t_half_max
    
    t_90_matrix = np.zeros(len(tot_random_t_90_low),5)
    t_90_matrix[:,0] = tot_random_t_90_med
    t_90_matrix[:,1] = tot_random_t_90_low
    t_90_matrix[:,2] = tot_random_t_90_high
    t_90_matrix[:,3] = tot_random_t_90_min
    t_90_matrix[:,4] = tot_random_t_90_max

    t_q_matrix = np.zeros(len(tot_random_t_q_low),5)
    t_q_matrix[:,0] = tot_random_t_q_med
    t_q_matrix[:,1] = tot_random_t_q_low
    t_q_matrix[:,2] = tot_random_t_q_high
    t_q_matrix[:,3] = tot_random_t_q_min
    t_q_matrix[:,4] = tot_random_t_q_max

    stats_matrix = np.zeros(len(coverage_list),7)
    stats_matrix[:,0] = coverage_list
    stats_matrix[:,0] = N_tot_mean
    stats_matrix[:,0] = N_tot_min
    stats_matrix[:,0] = N_tot_max
    stats_matrix[:,0] = [total_t_half for xx in coverage_list]
    stats_matrix[:,0] = [total_t_90 for xx in coverage_list]
    stats_matrix[:,0] = [total_t_q for xx in coverage_list]

    return t_half_matrix, t_90_matrix, t_q_matrix, stats_matrix

def calculate_sim_coverage_scatter(hdf5_file,coverage_list,N_trials):
    from random import sample
    from scipy.stats.mstats import chisquare
    import yt, h5py, re, os
    from simple_tools import high_low_limit
    import numpy as np
    from random import sample
    from scipy.stats.mstats import chisquare
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    a_bins = np.linspace(0.0,1.0,10000)
    T_bins = [cosmo.age(1.0/xx - 1.0) for xx in a_bins]
    T_bins_fix = [(T_bins[ii]+T_bins[ii+1])/2.0 for ii in range(len(T_bins)-1)]
    '''error as a function of coverage fraction'''

    '''fiducial scatter from SFH variation'''
    from random import sample
    from scipy.stats.mstats import chisquare

    SFH_data = h5py.File(hdf5_file)
    
    all_star_ages = []

    for los_key in SFH_data['LOS_data'].keys():
        '''dump all the particles into one list'''
        target_los = SFH_data['LOS_data'][los_key]['age'][:]
        [all_star_ages.append(xx) for xx in target_los]
        
    con_t_half_min, con_t_half_max, con_t_half_med, con_t_half_low, con_t_half_high = [],[],[],[],[]
    con_t_90_min, con_t_90_max, con_t_90_med, con_t_90_low, con_t_90_high = [],[],[],[],[]
    con_t_q_min, con_t_q_max, con_t_q_med, con_t_q_low, con_t_q_high = [],[],[],[],[]

    for ii in coverage_list:
        tot_t_50s, tot_t_90s, tot_t_qs = [],[],[]
        for jj in range(N_trials):
            randint_list = sample(xrange(1,100), ii)
            N_tot = 0
            constructed_sfh = [0.0 for xx in range(len(T_bins_fix))]
            constructed_sfh = np.asarray(constructed_sfh)
            for kk in randint_list:
                '''Now I need to grab the correct los'''
                target_los = SFH_data['LOS_data']['los_'+str(kk)]['age'][:]
                N_tot = N_tot + len(target_los)
                age_hist, age_bin = np.histogram(target_los,bins=a_bins,normed=False)
                c_age_hist = np.cumsum(age_hist) 
                constructed_sfh = constructed_sfh+np.asarray(c_age_hist)

            constructed_sfh_norm = constructed_sfh/float(max(constructed_sfh))
            con_t_half_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.5])
            con_t_90_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=0.9])
            con_t_q_list = min([T_bins_fix[xx] for xx in range(len(constructed_sfh_norm)) if constructed_sfh_norm[xx]>=1.0])
            tot_t_50s.append(con_t_half_list)
            tot_t_90s.append(con_t_90_list)
            tot_t_qs.append(con_t_q_list)

        low, high, med = high_low_limit(tot_t_50s,0.68)
        con_t_half_low.append(low)
        con_t_half_high.append(high)
        con_t_half_med.append(med)
        con_t_half_min.append(min(tot_t_50s))
        con_t_half_max.append(max(tot_t_50s))
    
        low, high, med = high_low_limit(tot_t_90s,0.68)
        con_t_90_low.append(low)
        con_t_90_high.append(high)
        con_t_90_med.append(med)
        con_t_90_min.append(min(tot_t_90s))
        con_t_90_max.append(max(tot_t_90s))
    
        low, high, med = high_low_limit(tot_t_qs,0.68)
        con_t_q_low.append(low)
        con_t_q_high.append(high)
        con_t_q_med.append(med)
        con_t_q_min.append(min(tot_t_qs))
        con_t_q_max.append(max(tot_t_qs))

    sim_t_half_matrix = np.zeros(len(con_t_half_low),6)
    sim_t_half_matrix[:,0] = coverage_list
    sim_t_half_matrix[:,1] = con_t_half_med
    sim_t_half_matrix[:,2] = con_t_half_low
    sim_t_half_matrix[:,3] = con_t_half_high
    sim_t_half_matrix[:,4] = con_t_half_min
    sim_t_half_matrix[:,5] = con_t_half_max

    sim_t_90_matrix = np.zeros(len(con_t_90_low),6)
    sim_t_90_matrix[:,0] = coverage_list
    sim_t_90_matrix[:,1] = con_t_90_med
    sim_t_90_matrix[:,2] = con_t_90_low
    sim_t_90_matrix[:,3] = con_t_90_high
    sim_t_90_matrix[:,4] = con_t_90_min
    sim_t_90_matrix[:,5] = con_t_90_max

    sim_t_q_matrix = np.zeros(len(con_t_q_low),6)
    sim_t_q_matrix[:,0] = coverage_list
    sim_t_q_matrix[:,1] = con_t_q_med
    sim_t_q_matrix[:,2] = con_t_q_low
    sim_t_q_matrix[:,3] = con_t_q_high
    sim_t_q_matrix[:,4] = con_t_q_min
    sim_t_q_matrix[:,5] = con_t_q_max
    
    return sim_t_half_matrix, sim_t_90_matrix, sim_t_q_matrix

def slice_plotter(los_cell_file,cell_list,z_dist_bins,R_half,center):
    #The idea is to take the slice plotting program and make it modular
    #I want to plot as a function of distance along the Z axis the following things
    #
    #1) age of the stars (average)
    #2) density of stars in the cell
    #3) contribution of the outer edges to each (maybe some sort of weighted density)
    #   
    # The idea is I want to know why the gradient in age seems so steep when
    # doing things in projection versus radially. The inital idea is that 
    # the "outskirts" of the galaxy contribute more as you get further away from the 
    # center of the galaxy in projection 
    import numpy as np
    import h5py, re, os
    from astropy.cosmology import FlatLambdaCDM
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    f = h5py.File(los_cell_file)

    los_numbers = cell_list
    
    cell_size = R_half/5.0

    age_list, rho_list, rho_norm_list = [], [], []
    
    dist_proj_list = []

    for jj in range(len(los_numbers)):
        particle_coordinates = f['LOS_data']['los_'+str(los_numbers[jj])]['coordinates'][:]
        particle_ages = f['LOS_data']['los_'+str(los_numbers[jj])]['age'][:]

        particle_ages_T = np.asarray([cosmo.age(1.0/xx - 1.0) for xx in particle_ages])

        part_X = particle_coordinates[:,0]
        part_Z = particle_coordinates[:,2]
        part_Y = particle_coordinates[:,1]

        part_dist_proj = np.sqrt((np.mean(part_X)-center[0])**2.0+(np.mean(part_Y)-center[1])**2.0)

        #plot these all on one and color by distance?

        X_dist = part_X - center[0]
        Z_dist =np.asarray(part_Z - center[2])

        age_bins_avg = []
        z_bins_avg = []
        rho_bins = []

        for ii in range(len(z_dist_bins)-1):
            bin_size = abs(z_dist_bins[ii]-z_dist_bins[ii+1])
            bin_vol = bin_size*cell_size**2.0
            z_dist_cut = (Z_dist>=z_dist_bins[ii])&(Z_dist<z_dist_bins[ii+1])
            age_select = particle_ages_T[z_dist_cut]
            age_bins_avg.append(np.mean(age_select))
            z_bins_avg.append((z_dist_bins[ii]+z_dist_bins[ii+1])/2.0)
            rho_bins.append(float(len(age_select))/bin_vol)
        
        rho_bins_norm = [xx/np.max(rho_bins) for xx in rho_bins]
        
        dist_proj_list.append(part_dist_proj)
        age_list.append(age_bins_avg)
        rho_list.append(rho_bins)
        rho_norm_list.append(rho_bins_norm)

    age_array = np.asarray(age_list)
    rho_array = np.asarray(rho_list)
    rho_norm_array = np.asarray(rho_norm_list)

    np.reshape(age_array,(len(age_array),len(age_array[0])))
    np.reshape(rho_array,(len(rho_array),len(rho_array[0])))
    np.reshape(rho_norm_array,(len(rho_norm_array),len(rho_norm_array[0])))

    return age_array, rho_array, rho_norm_array, dist_proj_list

def bin_finder_old(target_list,start,bin_size,step_size):
    ###############
    # this program takes a list (array) and along with a step size
    # dumps it into bins so that they have a uniform number of particles
    # but not necessarily uniform in space

    N_tot = len(target_list)
    N_bins = int(float(N_tot))/bin_size
    
    print 'The total number of star particles is: '+str(N_tot)+' so there should be (about) '+str(N_bins)+' bins'
    
    r_current = start
    N_part = 0
    final_radius_list = []
    final_radius_list.append(start)
    N_gathered = 0
    bin_number = 0
    #for bin_number in range(N_bins): old iterator
    while N_gathered != N_tot:
        step_size=step_size
        r_prev = r_current
        while N_part < bin_size:
            r_current = r_current+step_size
            proj_bin_mask = (target_list>r_prev)&(target_list<r_current)
            gathered_mask = (target_list<r_current)
            N_gathered = len(target_list[gathered_mask])
            N_part = len(target_list[proj_bin_mask])
        final_radius = r_current
        bin_number += 1
        print 'bin number: '+str(bin_number)
        print 'the bin is between '+str(r_prev)+' and '+str(final_radius)+' and contains '+str(N_part)+' particles.\n'
        N_part = 0
        #final_bin_mask = (target_list>r_prev)&(target_list<final_radius)
        final_radius_list.append(final_radius)
    return final_radius_list

def bin_finder(target_list,start,bin_size,step_size,r_gal):
    ###############
    # this program takes a list (array) and along with a step size
    # dumps it into bins so that they have a uniform number of particles
    # but not necessarily uniform in space

    N_tot = len(target_list)
    N_bins = int(float(N_tot))/bin_size
    
    print 'The total number of star particles is: '+str(N_tot)+' so there should be (about) '+str(N_bins)+' bins'
    
    r_current = start
    N_part = 0
    final_radius_list = []
    final_radius_list.append(start)
    N_gathered = 0
    bin_number = 0
    r_prev = 0
    number_of_steps = r_gal/step_size
    #for bin_number in range(N_bins): old iterator
    for step_number in range(int(number_of_steps)):
        r_current = r_current+step_size
        proj_bin_mask = (target_list>r_prev)&(target_list<r_current)
        N_part = len(target_list[proj_bin_mask])
        if N_part >= bin_size:
            final_radius = r_current
            total_bin_mask = (target_list<final_radius)
            bin_number += 1
            print 'bin number: '+str(bin_number)
            print 'the bin is between '+str(r_prev)+' and '+str(final_radius)+' and contains '+str(N_part)+' particles.\n'
            print 'the total number of particles is now: '+str(len(target_list[total_bin_mask]))
            r_prev = r_current
            N_part = 0
            final_radius_list.append(final_radius)
        else:
            continue

    return final_radius_list

def most_accurate_radius(hdf5_file,R_gal,R_half,center,radius_bins=None):
    import numpy as np
    import h5py, re, os
    from astropy.cosmology import FlatLambdaCDM
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #####
    # The purpose of this program is to find the radius at which 
    # the sfh is closest to the total SFH 
    # and do this both in projection or radially
    # Note this doesn't matter at all for the galaxy as a whole
    # at least as defined by the "galaxy radius" the SFH
    # for the total galaxy is basically the same in both
    # projection and radially
    #
    ############
    #
    # Inputs: 
    # hdf5_file - just the file with the star particles
    # stats_file - file with the halo statistics like the positions
    #              and galaxy radius, and such
    # radius_bins - the bins within which the sfhs should be choosen
    # 
    # outputs:
    # sfh_outputs - an array where the rows are the SFH for stars
    #               between radius_bins[ii] and radius_bins[ii+1]
    #               The length of this should be the length
    #               of the radius bins - 1
    # 
    # square_diff - the squared difference between the total SFH
    #               and the SFH in each bin, once again this
    #               has the length of radius_bins - 1
    #
    ################

    #####
    #
    # To do 
    # change mentions of proj_bins
    # make the radial bin sizes reflective of the actual size of the galaxy

    #make time list
    cosmo = FlatLambdaCDM(H0=71.0,Om0=0.266)

    h = 0.71
    center = center

    time_bins = np.linspace(0.0,13.7,1000)

    f = h5py.File(hdf5_file)
    star_coords = f['PartType4']['Coordinates'][:]
    star_mass = f['PartType4']['Masses'][:]
    star_age = f['PartType4']['StellarFormationTime'][:] #units are a
    star_ages_T = np.asarray([cosmo.age(1.0/xx - 1.0) for xx in star_age])

    part_X = star_coords[:,0]
    part_Z = star_coords[:,2]
    part_Y = star_coords[:,1]

    part_dist_proj = np.sqrt((part_X-center[0])**2.0+(part_Y-center[1])**2.0)
    star_dist = np.sqrt((part_X-center[0])**2.0+(part_Y-center[1])**2.0+(part_Z-center[2])**2.0)

    total_mask = (star_dist>0.0)&(star_dist<R_gal)
    star_age_T_tot = star_ages_T[total_mask]
    star_dist_gal = star_dist[total_mask]
    star_dist_proj_gal = part_dist_proj[total_mask]
    N_tot = len(star_age_T_tot)
    total_hist, total_bins = np.histogram(star_age_T_tot,bins=time_bins)
    total_hist_c = np.cumsum(total_hist)
    total_rad_hist_c_norm = total_hist_c/float(max(total_hist_c))

    r_half_mask = (star_dist>0.0)&(star_dist<R_half)
    star_age_T_half = star_ages_T[r_half_mask]
    star_dist_half = star_dist[r_half_mask]
    star_dist_proj_half = part_dist_proj[total_mask]
    N_half = len(star_age_T_half)
    r_half_hist, r_half_bins = np.histogram(star_age_T_half,bins=time_bins)
    r_half_hist_c = np.cumsum(r_half_hist)
    r_half_rad_hist_c_norm = r_half_hist_c/float(max(r_half_hist_c))
    
    R_list, R_list_proj = [], []
    square_diff_proj_list, T_histogram_proj_list = [], []
    square_diff_rad_list, T_histogram_rad_list = [], []
    square_diff_proj_c_list, T_histogram_proj_c_list = [], []
    square_diff_rad_c_list, T_histogram_rad_c_list = [], []

    if radius_bins==None:
        print 'starting bin finder'
        radius_bins = bin_finder(star_dist_gal,0.0,100,R_half/100.0,R_gal)

        print 'starting proj bin finder'

        radius_bins_proj = bin_finder(star_dist_proj_gal,0.0,100,R_half/100.0,R_gal)

    for ii in range(len(radius_bins)-1):
        R_list.append((radius_bins[ii]+radius_bins[ii+1])/2.0)
        #binned SFHs in projection
        dist_mask = (star_dist_gal>radius_bins[ii])&(star_dist_gal<radius_bins[ii+1])
        star_ages_T_select = star_age_T_tot[dist_mask]
        print 'between '+str(radius_bins[ii])+' kpc and '+str(radius_bins[ii+1])+'kpc'

        print 'radial: '
        print 'N_bin: '+str(len(star_ages_T_select))+', N_tot: '+str(N_tot)

        #binned SFHs radially
        radial_mask = (star_dist>radius_bins[ii])&(star_dist<radius_bins[ii+1])
        star_ages_T_select_radial = star_age_T_tot[radial_mask]

        segment_hist, segement_bins = np.histogram(star_ages_T_select,bins=time_bins)
        segment_hist_c = np.cumsum(segment_hist)
        segment_hist_c_norm = segment_hist_c/float(max(segment_hist_c))
        segment_diff = segment_hist_c_norm - total_rad_hist_c_norm
        square_diff_rad_list.append(np.linalg.norm(segment_diff)) 
        T_histogram_rad_list.append(segment_hist_c_norm)

        #cumulative binned SFHs radially
        radial_mask = (star_dist>0.0)&(star_dist<radius_bins[ii])
        star_ages_T_select_radial = star_ages_T[radial_mask]

        #print 'radial cumulative: '
        #print 'N_bin: '+str(len(star_ages_T_select_radial))+', N_tot: '+str(N_tot)

        segment_hist, segement_bins = np.histogram(star_ages_T_select,bins=time_bins)
        segment_hist_c = np.cumsum(segment_hist)
        segment_hist_c_norm = segment_hist_c/float(max(segment_hist_c))
        segment_diff = segment_hist_c_norm - total_rad_hist_c_norm
        square_diff_rad_c_list.append(np.linalg.norm(segment_diff)) 
        T_histogram_rad_c_list.append(segment_hist_c_norm)
    
    for jj in range(len(radius_bins_proj)-1):
        R_list_proj.append((radius_bins_proj[jj]+radius_bins_proj[jj+1])/2.0)
        #binned SFHs in projection
        dist_mask_proj = (star_dist_proj_gal>radius_bins_proj[jj])&(star_dist_proj_gal<radius_bins_proj[jj+1])
        star_ages_T_select = star_ages_T[dist_mask_proj]
        print 'between '+str(radius_bins[jj])+' kpc and '+str(radius_bins[jj+1])+'kpc'

        print 'projected: '
        print 'N_bin: '+str(len(star_ages_T_select))+', N_tot: '+str(N_tot)

        segment_hist, segement_bins = np.histogram(star_ages_T_select,bins=time_bins)
        segment_hist_c = np.cumsum(segment_hist)
        segment_hist_c_norm = segment_hist_c/float(max(segment_hist_c))
        segment_diff = segment_hist_c_norm - total_rad_hist_c_norm
        square_diff_proj_list.append(np.linalg.norm(segment_diff)) 
        T_histogram_proj_list.append(segment_hist_c_norm)

        proj_mask = (star_dist_proj_gal>0.0)&(star_dist_proj_gal<radius_bins_proj[jj])
        star_ages_T_select = star_ages_T[proj_mask]

        segment_hist, segement_bins = np.histogram(star_ages_T_select,bins=time_bins)
        segment_hist_c = np.cumsum(segment_hist)
        segment_hist_c_norm = segment_hist_c/float(max(segment_hist_c))
        segment_diff = segment_hist_c_norm - total_rad_hist_c_norm
        square_diff_proj_c_list.append(np.linalg.norm(segment_diff)) 
        T_histogram_proj_c_list.append(segment_hist_c_norm)

    T_histogram_proj_array = np.asarray(T_histogram_proj_list)
    np.reshape(T_histogram_proj_array,(len(T_histogram_proj_array),len(T_histogram_proj_array[0])))

    T_histogram_rad_array = np.asarray(T_histogram_rad_list)
    np.reshape(T_histogram_rad_array,(len(T_histogram_rad_array),len(T_histogram_rad_array[0])))

    T_histogram_proj_c_array = np.asarray(T_histogram_proj_c_list)
    np.reshape(T_histogram_proj_c_array,(len(T_histogram_proj_c_array),len(T_histogram_proj_c_array[0])))

    T_histogram_rad_c_array = np.asarray(T_histogram_rad_c_list)
    np.reshape(T_histogram_rad_c_array,(len(T_histogram_rad_c_array),len(T_histogram_rad_c_array[0])))

    return T_histogram_proj_array, T_histogram_rad_array, T_histogram_proj_c_array, T_histogram_rad_c_array, total_rad_hist_c_norm, r_half_rad_hist_c_norm, square_diff_proj_list, square_diff_rad_list, square_diff_proj_c_list, square_diff_rad_c_list, R_list, R_list_proj
