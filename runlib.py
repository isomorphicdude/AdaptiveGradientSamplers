import ast
from utils import * 
from generate import get_2D_single_sampler
from samplers.jax_samplers import check_sampler_name
import numpy as np



def get_best_hyperparams(
    method,
    distribution,
    num_samples,
    hyperparams_list,
    name_list,
    return_best_samps=False,
    metric='wass2d',
    use_num_cores=2,
    num_runs = 30,
    init_states = None
):
    """
    Gets the best hyperparameters from a list of hyperparameters.  
    
    Args:  
        - method: the method to use
        - distribution: a toy class object in toy_model.py
        - num_samples: the number of samples to generate
        - hyperparams_list: a list of hyperparameter lists e.g. [range1, range2, ...]
        - name_list: a list of names for each hyperparameter e.g. ['name1', 'name2', ...]
        - return_best_samps: whether to return the best samples
        - metric: the metric to use to determine the best hyperparameters, 'kl_div' or 'wass2d'
        - use_num_cores: the number of cores to use
        - num_runs: use the metric averaged over num_runs runs to determine the best hyperparameters
        - init_states: either None or a numpy array of shape (num_runs, 2), if None then the init_state 
            is randomly generated from N(0, 100)
        
        
    Returns:  
        - best_hyperparams: the best hyperparameters
        - best_samps: the best samples if return_best_samps is True
    """  
    
    
    _hyperparams_list = create_hyperparams_list(hyperparams_list,
                                                name_list=name_list)
    
    generate_2D_single = get_2D_single_sampler(method)
    
    samples_list =  []
    
    if init_states is None:
        init_states = np.random.normal(loc=0, scale=2, size=(num_runs, 2))
    
    for i in range(num_runs):
        init_state = init_states[i, :]
        samples = generate_2D_single(distribution=distribution,
                                    init_state = init_state,
                                    num_samples=num_samples,
                                    hyperparams=_hyperparams_list,
                                    use_num_cores=use_num_cores,
                                    seed = i)
        samples_list.append(samples)
    
    # recreate the dictionary from a list of dictionaries by unifying the same keys
    # the value should be a dictionary of lists, indexed by hyperparameters
    all_keys = list(samples_list[0].keys())
    l_samples = {key: [] for key in all_keys}
    
    for samp in samples_list:
        for key in all_keys:
            l_samples[key].append(samp[key])
    
    out = []
    
    if metric=='kl_div':
        kl_div_grid_search = {name: None for name in samples.keys()}
        for key in samples.keys():
            key_metric = np.mean([distribution.kl_divergence(samp) for samp in l_samples[key]])
            kl_div_grid_search[key] = key_metric
        
        best_hyp = ast.literal_eval(min(kl_div_grid_search,
                                        key=kl_div_grid_search.get))
           
        out.append(best_hyp)
        
        if return_best_samps:
            out.append(samples[min(kl_div_grid_search, key=kl_div_grid_search.get)])
            
    elif metric=='wass2d':
        w2d_grid_search = {name: None for name in samples.keys()}
        for key in samples.keys():
            key_metric = np.mean([distribution.wass2d(samp) for samp in l_samples[key]])
            w2d_grid_search[key] = key_metric
            
        # print(w2d_grid_search)
        try:
            best_hyp = ast.literal_eval(min(w2d_grid_search,
                                        key=w2d_grid_search.get))
        except:
            best_hyp = min(w2d_grid_search,
                            key=w2d_grid_search.get)
        out.append(best_hyp)
        
        if return_best_samps:
            out.append(samples[min(w2d_grid_search, 
                                   key=w2d_grid_search.get)])
    
    return out


def get_best_hyperparams_list(methods_list,
                              distribution,
                              num_samples,
                              hyperparams_list_dict,
                              name_list_dict,
                              return_best_samps=False,
                              metric='wass2d',
                              use_num_cores=2,
                              num_runs = 30,
                              init_states = None):
    """
    Produces a dictionary of dictionaries as hyperparameters
    
    Args:   
        - methods_list: a list of methods to use, list of strings
        - distribution: a toy class object in toy_model.py
        - num_samples: the number of samples to generate
        - hyperparams_list_dict: a dictionary of lists of hyperparameters
        - name_list_dict: a dictionary of lists of names for each hyperparameter
        - return_best_samps: whether to return the best samples
        - metric: the metric to use to determine the best hyperparameters, 'kl_div' or 'wass2d'
        - use_num_cores: the number of cores to use  
        - num_runs: use the metric averaged over num_runs runs to determine the best hyperparameters
        - init_states: either None or a numpy array of shape (num_runs, 2), if None then the init_state
        
    Returns:  
        - out: a dictionary of tuples (best hyperparameters, best samples if return_best_samps is True)
    """
    # check if methods are valid
    check_sampler_name(methods_list)
    
    out = {method: None for method in methods_list}
    for method in methods_list:
        out[method] = get_best_hyperparams(method=method,
                                           distribution=distribution,
                                           num_samples=num_samples,
                                           hyperparams_list=hyperparams_list_dict[method],
                                           name_list=name_list_dict[method],
                                           return_best_samps=return_best_samps,
                                           metric=metric,
                                           use_num_cores=use_num_cores,
                                           num_runs=num_runs,
                                           init_states=init_states)
    return out