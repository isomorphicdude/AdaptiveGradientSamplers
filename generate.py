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
                      key=key)[0]
    
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
        - samples: a dictionary of samples for each method and hyperparams
        - name_list: a dictionary of names for each method and hyperparams
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


sampler_ula = get_samplers('ula', fix=False)

def _ula_generate_2D_single(distribution,
                            init_state,
                            num_samples=10000,
                            hyperparams=[],
                            seed=0,
                            store=False,
                            output="ula.h5py",
                            use_num_cores = 1
                            ):
    """
    Returns results from multiple runs from ula on 2D distribution.  
    
    Args:   
        - distribution: a toy class object in toy_model.py
        - init_state: initial state for the chain
        - num_samples: number of samples to take per chain
        - method: method to run, string
        - hyperparams: list of hyperparameters, (list of dicts)
        - seed: random seed for jax.random
        - store: whether to store the samples in a file in h5py format
        - output: name of the output file, if store=True  
        - use_num_cores: number of cores to use for multiprocessing
        
    Returns:  
        - samples: a dictionary of samples indexed by hyperparameters
    """  
    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    # A dictionary of samples indexed by hyperparameters
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    # print(distribution.grad_logpdf)
    
    def sampler_fn(hyp):
        """Returns a sampler function with fixed hyperparameters"""
        # hyp is a dictionary of hyperparameters
        return sampler_ula(x0=init_state,
                            n_samples=num_samples,
                            grad_log_pdf = distribution.grad_logpdf,
                            eps=1e-5,
                            key=key,
                            **hyp)
        
    # import multiprocess after function so the workers 
    # can access the function
    import multiprocess
    from multiprocess import get_context

    cpu_count = multiprocess.cpu_count()
    
    if use_num_cores > cpu_count:
        print(f"Warning: using {cpu_count} cores instead of {use_num_cores}")
        use_num_cores = cpu_count
    
    with get_context('forkserver').Pool(use_num_cores) as p:
        out = p.map(sampler_fn, 
                    hyperparams,
                    chunksize=chunksize)
        
    # store and return (note map is order preserving)
    for i, hyp in enumerate(hyperparams):
        ret[f"{hyp}"] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret



sampler_pula = get_samplers('pula', fix=False)
def _pula_generate_2D_single(distribution,
                             init_state,
                             num_samples=10000,
                             hyperparams=[],
                             seed=0,
                             store=False,
                             output="pula.h5py",
                             use_num_cores = 1,
                             P = None
                             ):
    
    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_pula(x0=init_state,
                            n_samples=num_samples,
                            grad_log_pdf = distribution.grad_logpdf,
                            eps=1e-5,
                            key=key,
                            P = P,
                            **hyp)
        
    import multiprocess
    from multiprocess import get_context

    cpu_count = multiprocess.cpu_count()
    
    if use_num_cores > cpu_count:
        print(f"Warning: using {cpu_count} cores instead of {use_num_cores}")
        use_num_cores = cpu_count
    
    with get_context('forkserver').Pool(use_num_cores) as p:
        out = p.map(sampler_fn, 
                    hyperparams,
                    chunksize=chunksize)
        
    # store and return (note map is order preserving)
    for i, hyp in enumerate(hyperparams):
        ret[f"{hyp}"] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret


sampler_rmsprop = get_samplers('rmsprop', fix=False)

def _rmsprop_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="rmsprop.h5py",
                                use_num_cores = 1
                                ):

    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_rmsprop(x0=init_state,
                            n_samples=num_samples,
                            grad_log_pdf = distribution.grad_logpdf,
                            eps=1e-5,
                            key=key,
                            **hyp)
        
    import multiprocess
    from multiprocess import get_context

    cpu_count = multiprocess.cpu_count()
    
    if use_num_cores > cpu_count:
        print(f"Warning: using {cpu_count} cores instead of {use_num_cores}")
        use_num_cores = cpu_count
    
    with get_context('forkserver').Pool(use_num_cores) as p:
        out = p.map(sampler_fn, 
                    hyperparams,
                    chunksize=chunksize)
        
    # store and return (note map is order preserving)
    for i, hyp in enumerate(hyperparams):
        ret[f"{hyp}"] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret


sampler_rmsfull = get_samplers('rmsfull', fix=False)

def _rmsfull_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="rmsfull.h5py",
                                use_num_cores = 1
                                ):
    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_rmsfull(x0=init_state,
                            n_samples=num_samples,
                            grad_log_pdf = distribution.grad_logpdf,
                            eps=1e-5,
                            key=key,
                            **hyp)
        
    import multiprocess
    from multiprocess import get_context

    cpu_count = multiprocess.cpu_count()
    
    if use_num_cores > cpu_count:
        print(f"Warning: using {cpu_count} cores instead of {use_num_cores}")
        use_num_cores = cpu_count
    
    with get_context('forkserver').Pool(use_num_cores) as p:
        out = p.map(sampler_fn, 
                    hyperparams,
                    chunksize=chunksize)
        
    # store and return (note map is order preserving)
    for i, hyp in enumerate(hyperparams):
        ret[f"{hyp}"] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret



single_sampler_dict = {
    'ula': _ula_generate_2D_single,
    'pula': _pula_generate_2D_single,
    'rmsprop': _rmsprop_generate_2D_single,
    'rmsfull': _rmsfull_generate_2D_single,
}

def get_2D_single_sampler(method='ula'):
    return single_sampler_dict[method]