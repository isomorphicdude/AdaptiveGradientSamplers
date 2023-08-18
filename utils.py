"""Utility functions for the project."""
import h5py
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


def _get_name(name, full=True):
    """Gets the name of the sampler from the filename."""
    if not full:
        return name.split('_')[0]
    else:
        return f"{name.split('_')[0]}_{name.split('_')[-1]}"


def get_samples_dict(filename):
    """
    Returns a dictionary of samples from a h5py file.
    The keys are names and the last index.
    """
    ret = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            ret[_get_name(key)] = f[key][...]
    return ret



def kl_div_every_n(distribution, samples, n):
    """
    Compute the KL divergence every n samples.  
    
    Args:  
        - distribution: a toy class object in toy_model.py
        - samples: a dictionary of samples, keys are methods and hyperparameters
        - n: compute the KL divergence every n samples
        
    Returns:  
        - kl_div: a dictionary of KL divergences for each method and hyperparameter
    """
    kl_div = {f'{name.split("_")[0]}_{name.split("_")[-1]}': \
        [] for name in samples.keys()}
    
    for name in samples.keys():
        for i in range(n, len(samples[name]), n):
            kl_div[f'{name.split("_")[0]}_{name.split("_")[-1]}'].\
                append(distribution.kl_divergence(samples[name][:i]))
            
    return kl_div



def w2d_every_n(distribution, samples, n):
    """
    Compute the 2-Wasserstein distance every n samples.
    
    Args:   
        - distribution: a toy class object in toy_model.py
        - samples: a dictionary of samples, keys are methods and hyperparameters
        - n: compute the 2-Wasserstein distance every n samples
        
    Returns:    
        - w2d: a dictionary of 2-Wasserstein distances for each method and hyperparameter
    """
    w2d = {f'{name.split("_")[0]}_{name.split("_")[-1]}': \
        [] for name in samples.keys()}
    
    for name in samples.keys():
        for i in range(n, len(samples[name]), n):
            w2d[f'{name.split("_")[0]}_{name.split("_")[-1]}'].\
                append(distribution.wass2d(samples[name][:i]))
            
    return w2d


def create_hyperparams_list(list_of_hyperparams,
                            name_list):
    """
    Returns a list of hyperparameter dictionaries from a list of ranges.  
    
    Args:  
        - list_of_hyperparams: a list of ranges for each hyperparameter
        - name_list: a list of names for each hyperparameter, in the same order
        
    Returns:
        - hyperparams_list: a list of hyperparameter dictionaries,
        each has the form {hyp1: val1, hyp2: val2, ...}
    """  
    _list = product(*list_of_hyperparams)
    hyperparams_list = [dict(zip(name_list, x)) for x in _list]
    
    return hyperparams_list


def plot_2D_samples(samples,
                    distribution, 
                    xlim, 
                    ylim):
    plt.figure(figsize=(8, 6))
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap='RdBu')
    plt.
