import ast
from utils import * 
from generate import get_2D_single_sampler



def get_best_hyperparams(
    method,
    distribution,
    init_state,
    num_samples,
    hyperparams_list,
    name_list,
    return_best_samps=False,
    metric='wass2d',
    use_num_cores=2,
):
    """
    Gets the best hyperparameters from a list of hyperparameters.  
    
    Args:  
        - method: the method to use
        - distribution: a toy class object in toy_model.py
        - init_state: the initial state
        - num_samples: the number of samples to generate
        - hyperparams_list: a list of hyperparameter lists e.g. [range1, range2, ...]
        - name_list: a list of names for each hyperparameter e.g. ['name1', 'name2', ...]
        - return_best_samps: whether to return the best samples
        - metric: the metric to use to determine the best hyperparameters  
        - use_num_cores: the number of cores to use
        
    Returns:  
        - best_hyperparams: the best hyperparameters
        - best_samps: the best samples if return_best_samps is True
    """  
    
    
    
    _hyperparams_list = create_hyperparams_list(hyperparams_list,
                                                name_list=name_list)
    
    generate_2D_single = get_2D_single_sampler(method)
    
    samples = generate_2D_single(distribution=distribution,
                                 init_state = init_state,
                                 num_samples=num_samples,
                                 hyperparams=_hyperparams_list,
                                 use_num_cores=use_num_cores,)
    
    out = []
    
    if metric=='kl_div':
        kl_div_grid_search = {name: None for name in samples.keys()}
        for key in samples.keys():
           kl_div_grid_search[key] = distribution.kl_divergence(samples[key])
        
        best_hyp = ast.literal_eval(min(kl_div_grid_search,
                                        key=kl_div_grid_search.get))
           
        out.append(best_hyp)
        
        if return_best_samps:
            out.append(samples[min(kl_div_grid_search, key=kl_div_grid_search.get)])
            
    elif metric=='wass2d':
        w2d_grid_search = {name: None for name in samples.keys()}
        for key in samples.keys():
            w2d_grid_search[key] = distribution.wass2d(samples[key])
        
        best_hyp = ast.literal_eval(min(w2d_grid_search,
                                        key=w2d_grid_search.get))
        out.append(best_hyp)
        
        if return_best_samps:
            out.append(samples[min(w2d_grid_search, 
                                   key=w2d_grid_search.get)])
    
    return out


def get_best_hyperparams_list(methods_list,
                              distribution,
                              init_state,
                              num_samples,
                              hyperparams_list_dict,
                              name_list_dict,
                              return_best_samps=False,
                              metric='wass2d',
                              use_num_cores=2):
    """
    Produces a dictionary of dictionaries as hyperparameters
    
    Args:   
        - methods_list: a list of methods to use, list of strings
        - distribution: a toy class object in toy_model.py
        - init_state: the initial state
        - num_samples: the number of samples to generate
        - hyperparams_list_dict: a dictionary of lists of hyperparameters
        - name_list_dict: a dictionary of lists of names for each hyperparameter
        - return_best_samps: whether to return the best samples
        - metric: the metric to use to determine the best hyperparameters
        - use_num_cores: the number of cores to use  
        
    Returns:  
        - out: a dictionary of tuples (best hyperparameters, best samples if return_best_samps is True)
    """
    out = {method: None for method in methods_list}
    for method in methods_list:
        out[method] = get_best_hyperparams(method=method,
                                           distribution=distribution,
                                           init_state=init_state,
                                           num_samples=num_samples,
                                           hyperparams_list=hyperparams_list_dict[method],
                                           name_list=name_list_dict[method],
                                           return_best_samps=return_best_samps,
                                           metric=metric,
                                           use_num_cores=use_num_cores)
    return out