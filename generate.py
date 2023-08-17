import os
import jax
import jax.numpy as jnp

import h5py

from functools import partial
from tqdm import tqdm

from toy_models import *
from samplers.jax_samplers import get_samplers


    
def run_single_chain(sampler, 
                     key,
                     init_state,
                     distribution, 
                     num_samples):
    
    samples = sampler(init_state, 
                      n_samples=num_samples,
                      grad_log_pdf = distribution.grad_logpdf,
                      key=key)
    
    return samples


def generate_2D(distribution, 
                num_chains=5,
                num_samples=10000,
                methods=['ula', 'rmsprop', 'rmsfull', 'pula'],
                hyperparams={},
                seed=0,
                store=False,
                output="data.h5py"
                ):
    """
    Run experiments to sample from a 2D Gaussian.  
    
    Args:  
        - distribution: a toy class object in toy_model.py
        - num_chains: number of chains to run, initialized with a Gaussian (0, I)
        - num_samples: number of samples to take per chain
        - methods: list of methods to run
        - hyperparams: dictionary of hyperparameters for each method (dict of dicts)
        - seed: random seed for jax.random  
        - store: whether to store the samples in a file in h5py format
        - output: name of the output file, if store=True
        
    Returns:   
        - samples: a dictionary of samples for each method and step size
    """  
    #TODO: add support for multiprocessing
    key = jax.random.PRNGKey(seed)
    init_states = jax.random.normal(key, shape=(num_chains, 2))
    
    # when not storing, only return the dictionary
    ret = {name: {} for name in methods}
    name_list = {name: [] for name in methods}
    
    for name in methods:
        sampler = get_samplers(name, fix=True, hyperparam=hyperparams.get(name))
        for i, init_state in enumerate(init_states):
            samples = run_single_chain(sampler, 
                                       key, 
                                       init_state, 
                                       distribution, 
                                       num_samples)
                
            ret[name][f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}"] = samples
            name_list[name].append(f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}")
            
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for name in methods:
                for key in name_list[name]:
                    f.create_dataset(key, 
                                     data=ret[name][key])
            
    return ret, name_list