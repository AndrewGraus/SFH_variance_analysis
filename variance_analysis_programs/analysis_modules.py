#The idea behind this is to make a package with all the modules I need
#I think to start I only really need two
#
#1) Port the program that calculates the SFH variance as a function of coverage fraction
#2) Port the program that bootstraps that variance calculation

def SFH_variance(hdf5_file,coverage_bins=[1,5,10,20,30,40,50,60,70,80,90,99]):
    from random import sample
    from scipy.stats.mstats import chisquare

    #SFH_variance
    #
    #This program takes a hdf5 file generated dividing the stellar distribution into different LOS
    #and generates the variance as a function of coverage fraction
    #
    #INPUTS:
    # hdf5_file - the file to do the analysis on
    # coverage_bins - the percentages to do the calculation on

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
        for jj in range(100):
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
                constructed_sfh = constructed_sfh+np.asarray(c_age_hist)
            #[constructed_sfh+np.asarray(small_segs[xx]) for xx in randint_list]
            if max(constructed_sfh) == 0.0:
                constructed_sfh_norm = [0.0 for xx in range(len(T_bins_fix))]
            else:
                constructed_sfh_norm = constructed_sfh/float(max(constructed_sfh))
                #print constructed_sfh_norm
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
        
    #assert len(rms_err_min) == len(range(99))
    '''plt.figure(1,(10,10))
    rc('axes',linewidth=3)
    plt.yticks(fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.xlabel('percent coverage',fontsize=30)
    plt.ylabel('difference?',fontsize=30)
    #print range(99), chi_squared_min
    plt.fill_between(range(99),con_t_half_min,con_t_half_max,color='b',alpha=0.4)
    plt.plot(range(99),con_t_half_mean,color='b',linestyle='--',linewidth=3)
    plt.ylim([0.0,14.0])
    plt.xlim([0.0,100.0])
    plt.show()'''
    
    con_t_half_err_low = [(con_t_half_low[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_low))]
    con_t_half_err_high = [(con_t_half_high[xx]-con_t_half_mean[xx])/con_t_half_mean[xx] for xx in range(len(con_t_half_high))]

    con_t_q_err_low = [(con_t_q_low[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_low))]
    con_t_q_err_high = [(con_t_q_high[xx]-con_t_q_mean[xx])/con_t_q_mean[xx] for xx in range(len(con_t_q_high))]
    
    con_t_90_err_low = [(con_t_90_low[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_low))]
    con_t_90_err_high = [(con_t_90_high[xx]-con_t_90_mean[xx])/con_t_90_mean[xx] for xx in range(len(con_t_90_high))]
    
    plt.figure(1,(20,10))
