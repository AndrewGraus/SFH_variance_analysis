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
