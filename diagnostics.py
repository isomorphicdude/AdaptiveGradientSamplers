"""Functions for evaluating MCMC convergence and mixing."""  

import numpy as np
import ot
from tqdm import tqdm
#TODO: use jax.numpy instead of numpy

# computing autocorrelation by hand
def acf(x, lags=None):
    """
    Computes the autocorrelation function.  
    
    Args:  
        - x: time series data, one dimensional array
        - lags: number of lags to compute autocorrelation for
        
    Returns:  
        - array of autocorrelation values, length equals lags
    """  
    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean
    corr = np.correlate(xp, xp, mode='full')[len(x)-1:]
    acf = corr/var
    acf = acf / len(x)
    
    if lags is not None:
        return acf[:lags]
    else:
        return acf
    
    
def ess(samples, acf_val=None):
    """
    Computes the effective sample size.  
    
    Args:  
        - samples: time series data, one dimensional array
        - acf_val: array of autocorrelation values, default None
        
    Returns:  
        - effective sample size
    """  
    if acf_val is None:
        acf_val = acf(samples)
    print(acf_val.sum())
    
    return 1/(1+2*np.sum(acf_val[1:]))


def emd2_uniform(samples, ground_truth):
    """Computes Wasserstein-2 distance between samples and ground truth."""
    M = ot.dist(samples, ground_truth, metric='euclidean', p=2)
    wass = ot.lp.emd2([], [], M)
    return wass


def emd2_mass(samples, ground_truth, bins=100):
    """Computes the Wasserstein-2 distace between samples and ground truth.
       Uses number of bins to create histograms. 
    """
    hist1, x1, y1 = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
    hist2, x2, y2 = np.histogram2d(ground_truth[:, 0], ground_truth[:,1], bins=bins, density=True)
    
    xx1, yy1 = np.meshgrid(x1[:-1], y1[:-1])
    xx2, yy2 = np.meshgrid(x2[:-1], y2[:-1])
    
    M = ot.dist(np.c_[xx1.ravel(), yy1.ravel()],
                np.c_[xx2.ravel(), yy2.ravel()],
                metric='euclidean', p=2)
    
    # normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    wass = ot.lp.emd2(hist1.ravel(), hist2.ravel(), M)
    
    return wass
    

def plot_emd2_uniform(samples1, 
                      samples2, 
                      ground_truth,
                      use_full=False,
                      max_n=5000, 
                      step=100):
    """
    Plots Wasserstein-2 distance between samples and ground truth for 
    two sets of samples.
    """
    wass_dist_list1 = []	
    wass_dist_list2 = []
    
    for i in tqdm(range(1, max_n, step), total=len(range(1,max_n,step)),
                  colour='green', desc='>>'):
        if use_full:
            wass_dist1 = emd2_uniform(samples1[:i], ground_truth)
            wass_dist2 = emd2_uniform(samples2[:i], ground_truth)
        else:
            wass_dist1 = emd2_uniform(samples1[:i], ground_truth[:i])
            wass_dist2 = emd2_uniform(samples2[:i], ground_truth[:i])
        wass_dist_list1.append(wass_dist1)
        wass_dist_list2.append(wass_dist2)
        
    return wass_dist_list1, wass_dist_list2

def plot_emd2_mass(samples1,
                   samples2,
                   ground_truth,
                   use_full=False,
                   max_n=5000,
                   step=100,
                   bins=100):
    """
    Plots Wasserstein-2 distance between samples and ground truth for
    two sets of samples.
    """
    wass_dist_list1 = []	
    wass_dist_list2 = []
    for i in tqdm(range(1, max_n, step), total=len(range(1,max_n,step)),
                  colour='green', desc='>>'):
        if use_full:
            wass_dist1 = emd2_mass(samples1[:i], ground_truth, bins=bins)
            wass_dist2 = emd2_mass(samples2[:i], ground_truth, bins=bins)
        else:
            wass_dist1 = emd2_mass(samples1[:i], ground_truth[:i], bins=bins)
            wass_dist2 = emd2_mass(samples2[:i], ground_truth[:i], bins=bins)
        wass_dist_list1.append(wass_dist1)
        wass_dist_list2.append(wass_dist2)
        
    return wass_dist_list1, wass_dist_list2


def discrete_kl(samples, 
                ground_truth, 
                pdf, 
                **kwargs):
    """Computes discrete KL divergence between samples and ground truth."""
    samples_alt = samples
    m1 = pdf(samples_alt, **kwargs) / pdf(samples_alt, **kwargs).sum()
    m2 = pdf(ground_truth, **kwargs) / pdf(ground_truth, **kwargs).sum()
    kl = np.sum(m1 * (np.log(m1) - np.log(m2)))
    return kl
    
def plot_kl(samples1, 
            samples2, 
            ground_truth,
            pdf,
            same_length=False,
            max_n=5000, 
            step=100,
            **kwargs):
    """
    Plots discrete KL divergence between samples and ground truth for 
    two sets of samples.
    """
    if not same_length:
        samples1_plt = samples1[1:]
        samples2_plt = samples2[1:]
    else:
        samples1_plt = samples1
        samples2_plt = samples2
        
    kl_dist_list1 = []	
    kl_dist_list2 = []
    
    for i in range(1, max_n, step):
        kl_dist1 = discrete_kl(samples1_plt[:i], ground_truth[:i], pdf, **kwargs)
        kl_dist2 = discrete_kl(samples2_plt[:i], ground_truth[:i], pdf, **kwargs)
        kl_dist_list1.append(kl_dist1)
        kl_dist_list2.append(kl_dist2)
        
    return kl_dist_list1, kl_dist_list2
    

                            