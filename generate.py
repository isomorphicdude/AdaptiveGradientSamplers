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
                     num_samples,
                     name=None
                     ):
    
    if name == 'adam':
        samples = sampler(init_state, 
                        n_samples=num_samples,
                        grad_log_pdf = distribution.grad_logpdf,
                        key=key,
                        eps=1e-8)[0]
    else:
        samples = sampler(init_state, 
                        n_samples=num_samples,
                        grad_log_pdf = distribution.grad_logpdf,
                        key=key,
                        eps=1e-5)[0]
    
    return samples


def generate_2D(distribution, 
                num_chains=5,
                num_samples=10000,
                methods=['ula', 'rmsprop', 'rmsfull', 'pula'],
                hyperparams={},
                seed=0,
                store=False,
                output="data.h5py",
                use_string_keys=True,
                init_var = 1.0,
                fixed_init_state = None,
                ):
    """
    Run experiments to sample from a 2D distribution.  
    
    Args:  
        - distribution: a toy class object in toy_model.py
        - num_chains: number of chains to run, initialized with a Gaussian (0, I)
        - num_samples: number of samples to take per chain
        - methods: list of methods to run
        - hyperparams: dictionary of hyperparameters for each method (dict of dicts)
        - seed: random seed for jax.random  
        - store: whether to store the samples in a file in h5py format
        - output: name of the output file, if store=True
        - use_string_keys: whether to use strings of hyperparams as keys
        - init_var: variance of the initial state
        - fixed_init_state: if not None, use this as the initial state for all chains
        
    Returns:   
        - samples: a dictionary of samples for each method and hyperparams
        - name_list: a dictionary of names for each method and hyperparams
    """  
    #TODO: add support for multiprocessing
    key = jax.random.PRNGKey(seed)
    init_states = jax.random.normal(key, shape=(num_chains, 2)) * jnp.sqrt(init_var)
    
    # when not storing, only return the dictionary
    ret = {}
    name_list = []
    
    # add P to the hyperparams if pula and Gaussian2D
    # otherwise raise error
    if 'pula' in methods and distribution.__class__.__name__=='Gaussian2D':
        # print(hyperparams['pula'])
        hyperparams['pula']['P'] = distribution.sigma
    elif 'pula' in methods and distribution.__class__.__name__!='Gaussian2D':
        raise ValueError("PULA only works for Gaussian2D")
    
    for name in methods:
        sampler = get_samplers(name, 
                               fix=True, 
                               hyperparam=hyperparams.get(name))
        if fixed_init_state is None:
            for i, init_state in enumerate(init_states):
                samples = run_single_chain(sampler, 
                                        key, 
                                        init_state, 
                                        distribution, 
                                        num_samples,
                                        name=name)
                
                if use_string_keys:
                    # name in method
                    ret[f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}"] = samples                
                    name_list.append(f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}")
        else:
            for i, init_state in enumerate(init_states):
                samples = run_single_chain(sampler, 
                                        key, 
                                        fixed_init_state, 
                                        distribution, 
                                        num_samples,
                                        name=name)
                if use_string_keys:
                    # name in method
                    ret[f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}"] = samples                
                    name_list.append(f"{name}_{hyperparams.get(name).values()}_{init_state}_{i}")
            
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in name_list:
                f.create_dataset(key, 
                                data=ret[key])
            
    return ret, name_list


sampler_ula = get_samplers('ula', fix=False)

def _ula_generate_2D_single(distribution,
                            init_state,
                            num_samples=10000,
                            hyperparams=[],
                            seed=0,
                            store=False,
                            output="ula.h5py",
                            use_num_cores = 1,
                            use_string_keys=True,
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
        - use_string_keys: whether to use strings of hyperparams as keys
        
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
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
                             P = None,
                             use_string_keys=True,
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
                            P = distribution.sigma,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret


sampler_adam_ula = get_samplers('adam', fix=False)

def _adam_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="adam.h5py",
                                use_num_cores = 1,
                                use_string_keys=True,
                                ):

    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_adam_ula(x0=init_state,
                            n_samples=num_samples,
                            grad_log_pdf = distribution.grad_logpdf,
                            eps=1e-8,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
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
                                use_num_cores = 1,
                                use_string_keys=True,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
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
                                use_num_cores = 1,
                                use_string_keys=True,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret



sampler_monge = get_samplers('monge', fix=False)

def _monge_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="monge.h5py",
                                use_num_cores = 1,
                                use_string_keys=True,
                                ):

    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_monge(x0=init_state,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret


sampler_forgetful_rmsprop = get_samplers('forgetful1', fix=False)

def _forgetful_rmsprop_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="forgetful_rmsprop.h5py",
                                use_num_cores = 1,
                                use_string_keys=True,
                                ):

    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_forgetful_rmsprop(x0=init_state,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
    # storing to h5py file
    if store:
        with h5py.File(output, 'w') as f:
            for key in ret.keys():
                f.create_dataset(key, 
                                 data=ret[key])
    
    return ret



sampler_forgetful_rmsprop2 = get_samplers('forgetful2', fix=False)

def _forgetful_rmsprop2_generate_2D_single(distribution,
                                init_state,
                                num_samples=10000,
                                hyperparams=[],
                                seed=0,
                                store=False,
                                output="forgetful_rmsprop2.h5py",
                                use_num_cores = 1,
                                use_string_keys=True,
                                ):

    num_runs = len(hyperparams)
    key = jax.random.PRNGKey(seed)
    
    chunksize = int(num_runs/(use_num_cores*10)) \
        if int(num_runs/(use_num_cores*10))>0 else 1
    
    ret = {f"{hyperparam}": None for hyperparam in hyperparams}
    
    
    def sampler_fn(hyp):
        return sampler_forgetful_rmsprop2(x0=init_state,
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
        if use_string_keys:
            ret[f"{hyp}"] = out[i][0]
        else:
            ret[hyp] = out[i][0]
    
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
    'adam': _adam_generate_2D_single,
    'monge': _monge_generate_2D_single,
    'forgetful1': _forgetful_rmsprop_generate_2D_single,
    'forgetful2': _forgetful_rmsprop2_generate_2D_single,
}

def get_2D_single_sampler(method='ula'):
    return single_sampler_dict[method]