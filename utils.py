"""Utility functions for the project."""
import jax
import jax.numpy as jnp
import numpy as np
import ast
import h5py
import pickle
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



def kl_div_every_n(distribution,
                   samples,
                   n,
                   use_stored=False):
    """
    Computes the KL divergence every n samples.  
    
    Args:  
        - distribution: a toy class object in toy_model.py
        - samples: a dictionary of samples, keys are methods and hyperparameters
        - n: compute the KL divergence every n samples
        - use_stored: whether to use the stored KL divergences
        
    Returns:  
        - kl_div: a dictionary of KL divergences for each method and hyperparameter
    """  
    
    kl_div = {f'{name.split("_")[0]}_{name.split("_")[-1]}': \
        [] for name in samples.keys()}
    
    for name in samples.keys():
        for i in range(n, len(samples[name])+1, n):
            kl_div[f'{name.split("_")[0]}_{name.split("_")[-1]}'].\
                append(distribution.kl_divergence(samples[name][:i]))
    return kl_div



def w2d_every_n(distribution, samples, n):
    """
    Computes the 2-Wasserstein distance every n samples.
    
    Args:   
        - distribution: a toy class object in toy_model.py
        - samples: a dictionary of samples, keys are methods and hyperparameters
        - n: compute the 2-Wasserstein distance every n samples
        
    Returns:    
        - w2d: a dictionary of 2-Wasserstein distances for each method and hyperparameter
    """
    
    if isinstance(samples, dict):
        w2d = {f'{name.split("_")[0]}_{name.split("_")[-1]}': \
            [] for name in samples.keys()}
        
        for name in samples.keys():
            for i in range(n, len(samples[name])+1, n):
                w2d[f'{name.split("_")[0]}_{name.split("_")[-1]}'].\
                    append(distribution.wass2d(samples[name][:i]))
            
        return w2d
    
    elif isinstance(samples, np.ndarray) or isinstance(samples, jnp.ndarray):
        w2d = []
        for i in range(n, len(samples)+1, n):
            w2d.append(distribution.wass2d(samples[:i]))
        return w2d
    
    

def get_samples_from_generated(generated_samples):
    """Replace the keys by simpler names."""
    ret = {f'{name.split("_")[0]}_{name.split("_")[-1]}': generated_samples[name] \
        for name in generated_samples.keys()}

    return ret
    
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


def _create_forget_times(interval_size, max_forget_time):
    """
    Returns a list of forget times.
    
    Args:  
        - interval_size: the size of the interval
        - max_forget_time: the maximum forget time
        
    Returns:
        - forget_times: a list of forget times
    """  
    forget_times = [i for i in range(0, max_forget_time, interval_size)]
    if forget_times[-1] != max_forget_time:
        forget_times.append(max_forget_time)
    return jnp.array(forget_times)

def get_forget_times(interval_size_list, max_forget_time_list):
    """
    Returns a list of lists of forget times.
    
    Args:  
        - interval_size_list: a list of interval sizes
        - max_forget_time_list: a list of maximum forget times
        
    Returns:
        - forget_times_list: a list of lists of forget times
    """  
    forget_times_list = []
    for interval_size, max_forget_time in zip(interval_size_list,
                                              max_forget_time_list):
        forget_times_list.append(_create_forget_times(interval_size,
                                                      max_forget_time))
    return forget_times_list


def plot_2D_samples_single(ax,
                           distribution,
                           samples,
                           first_n=-1,
                           xlim=3,
                           ylim=3,
                           **kwargs
                           ):
    """
    Plots 2D samples from a distribution and combine with density plot (if available).
    
    Args:  
        - ax: a matplotlib axis object
        - distribution: a toy class object in toy_model.py
        - samples: samples from a single method
        - first_n: plot the first n samples
        - xlim: the x limit
        - ylim: the y limit
        - **kwargs: keyword arguments for plotting samples  
        
    Returns:
        - ax: a matplotlib axis object
    """  
    distribution.plot(ax=ax, xlim=xlim, ylim=ylim, **kwargs)
    ax.scatter(samples[:first_n, 0], samples[:first_n, 1], s=3,
                    alpha=0.1, color='yellow')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    
    
def save_hyperparams(hyperparams_dict, path_to_save='hyperparams.txt'):
    """
    Saves the hyperparameters to a txt file.
    
    Args:  
        - hyperparams_dict: a dictionary of hyperparameters
        - path_to_save: the path to save the hyperparameters
    """  
    # with open(path_to_save, 'w') as f:
    #     print(f'Saving hyperparameters to {path_to_save}')
    #     f.write(str(hyperparams_dict))
    _save_hyperparams(hyperparams_dict, path_to_save=path_to_save)
        
def _save_hyperparams(hyperparams_dict, path_to_save='hyperparams.txt'):
    """Uses pickle to save the hyperparameters."""
    with open(path_to_save, 'wb') as f:
        print(f'Saving hyperparameters to {path_to_save}')
        pickle.dump(hyperparams_dict, f)
            

def _read_hyperparams(path_to_save='hyperparams.txt'):
    """Uses pickle to read the hyperparameters."""
    with open(path_to_save, 'rb') as f:
        print(f'Reading hyperparameters from {path_to_save}')
        data = pickle.load(f)
        f.close()
    return data


def read_hyperparams(path_to_save='hyperparam.txt'):
    """
    Returns a dictionary of hyperparameters from a txt file. 
    
    Args:  
        - path_to_save: the path to save the hyperparameters
    """   
    # with open(path_to_save, 'r') as f:
    #     data = f.read()
    #     f.close()
    # return eval(data)
    return _read_hyperparams(path_to_save=path_to_save)
        

    
    
    
    
    
    
