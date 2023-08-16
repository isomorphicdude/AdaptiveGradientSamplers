""""""
import jax
import numpy as np
import scipy
from functools import partial


def ula_sampler(x0, 
                n_samples,
                step_size, 
                grad_log_density,
                burnin=None,
                V_list=None,
                clip=False,
                eps=1e-5,
                **kwargs):
    """
    Returns samples from unadjusted langevin algorithm.  
    
    Args:  
        - x0: initial position
        - n_samples: number of samples to return
        - step_size: step size of algorithm
        - grad_log_density: gradient of log density, function
        - burnin: number of samples to discard
        - V_list: list of V values for preconditioning
        - clip: whether to clip the V values
        - eps: small constant to control extreme values
        - kwargs: additional arguments to grad_log_density  
        
    Returns:  
        - samples: samples from ULA 
    """
    x = x0
    samples = np.zeros((n_samples+1, 2))
    samples[0] = x0
    
    if V_list is None:
        for i in range(n_samples):
            x = x + step_size * grad_log_density(x, **kwargs) + \
                np.sqrt(2 * step_size) * np.random.normal(size=2)
            samples[i+1] = x
            
    else:
        print("V_list is not None, preconditioning is on.")
        for i in range(n_samples):
            V = V_list[i]
            if not clip:
                lr = step_size / (np.sqrt(V) + eps) # rmsprop way
            else:
                lr = step_size / np.clip(np.sqrt(V), a_min=eps, a_max=None) # clip way
            x = x + lr * grad_log_density(x, **kwargs) + np.sqrt(2 * lr) * np.random.normal(size=2)
            samples[i+1] = x
    
    if burnin is not None:
        samples = samples[burnin:]
        
    return samples


def rmsprop_ula(x0, 
                n_samples, 
                step_size, 
                beta,
                grad_log_density, 
                burnin=None, 
                eps=1e-4,
                get_V=False,
                get_grad=False,
                clip=False,
                **kwargs):
    """
    Implements the RMSprop ULA algorithm.  
    Args:  
        - x0: initial state
        - n_samples: number of samples to generate
        - step_size: step size
        - beta: decay parameter
        - grad_log_density: gradient of log density
        - burnin: number of samples to discard
        - eps: small constant to control extreme values
        - get_V: whether to return the V values
        - get_grad: whether to return the gradient values
        - clip: whether to clip the V values
        - kwargs: keyword arguments for grad_log_density  
        
    Returns:   
        - dictionary of samples, V values, and gradient values
    """
    x = x0
    samples = np.zeros((n_samples+1, 2))
    samples[0] = x0
    V_list = []
    grad_list = []
    V = 0
    
    for i in range(n_samples):
        g = grad_log_density(x, **kwargs)
        V = beta * V + (1-beta) * g**2
        if not clip:
            c = (np.sqrt(V)+ eps) # rmsprop way
        else:
            c = np.clip(np.sqrt(V), a_min=eps, a_max=None) # clip way
            
        x = x + step_size * (g/c) + np.sqrt(2 * step_size / c) * np.random.normal(size=2)
        samples[i+1] = x
        
        if get_V:
            V_list.append(V)
        if get_grad:
            grad_list.append(g)
        
    if burnin is not None:
        samples = samples[burnin:]
        
    out = {'samples': samples}
    
    if get_V:
        out['V'] = np.asanyarray(V_list)
    if get_grad:
        out['grad'] = np.asanyarray(grad_list)
    
    return out



def preconditioned_ula(x0, 
                n_samples,
                step_size, 
                grad_log_density,
                mat,
                burnin=None,
                **kwargs):
    """
    Returns samples from ULA using a fixed pre-condition matrix.
    
    Args:  
        - x0: initial position
        - n_samples: number of samples to return
        - step_size: step size of algorithm
        - grad_log_density: gradient of log density, function
        - mat: pre-condition matrix
        - burnin: number of samples to discard
        - kwargs: additional arguments to grad_log_density  
        
    Returns:  
        - samples: samples from ULA 
    """
    x = x0
    samples = np.zeros((n_samples+1, 2))
    samples[0] = x0
    
    for i in range(n_samples):
        x = x + step_size * mat @ grad_log_density(x, **kwargs)\
            + scipy.linalg.sqrtm(mat) @ np.random.normal(size=2) * np.sqrt(2 * step_size)
        samples[i+1] = x
        
    if burnin is not None:
        samples = samples[burnin:]
        
    return samples



# def rmsprop_ula2(x0, 
#                 n_samples, 
#                 step_size, 
#                 beta,
#                 grad_log_density, 
#                 burnin=None, 
#                 eps=1e-5,
#                 get_V=False,
#                 get_grad=False,
#                 get_gamma=False,
#                 **kwargs):
#     """
#     Implements the RMSprop ULA algorithm.  
#     Args:  
#         - x0: initial state
#         - n_samples: number of samples to generate
#         - step_size: step size
#         - beta: decay parameter
#         - grad_log_density: gradient of log density
#         - burnin: number of samples to discard
#         - eps: small constant to control extreme values
#         - get_V: whether to return the V values
#         - get_grad: whether to return the gradient values
#         - get_gamma: whether to return the Gamma values
#         - kwargs: keyword arguments for grad_log_density  
        
#     Returns:   
#         - dictionary of samples, V values, and gradient values
#     """
#     x = x0
#     samples = np.zeros((n_samples+1, 2))
#     samples[0] = x0
#     V_list = []
#     grad_list = []
#     V = 0
#     prev_V = 0
#     grad_G = 0
#     Gamma_list = []
    
#     grad_log_density_fn = partial(grad_log_density, **kwargs)
#     hess_grad_log_density = jax.jacfwd(grad_log_density_fn)
    
#     for i in range(n_samples):
#         grad = grad_log_density(x, **kwargs)
#         V = beta * V + (1-beta) * grad**2
#         G = (np.sqrt(V) + eps)
        
#         lr = step_size / G # rmsprop way
        
#         x = x + lr * grad +  2 * step_size * (1/G) * grad_G * (1/G)\
#             + step_size * (1/G) * (prev_V - V) * grad\
#             + np.sqrt(2 * lr) * np.random.normal(size=2)
        
#         grad_G = (1-beta) * (hess_grad_log_density(x).sum(axis=1))
#         # print(Gamma)
        
#         samples[i+1] = x
#         if get_V:
#             V_list.append(V)
#         if get_grad:
#             grad_list.append(grad)
#         if get_gamma:
#             Gamma_list.append(grad_G)
        
#     if burnin is not None:
#         samples = samples[burnin:]
        
#     out = {'samples': samples}
    
#     if get_V:
#         out['V'] = np.asanyarray(V_list)
#     if get_grad:
#         out['grad'] = np.asanyarray(grad_list)
#     if get_gamma:
#         out['gamma'] = np.asanyarray(Gamma_list)
    
#     return out