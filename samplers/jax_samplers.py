"""Jax implementation"""

import inspect
import numpy as np
import jax
import jax.numpy as jnp
from jax import config 
config.update("jax_debug_nans", True)
from functools import partial

    
    
########### original unadjusted langevin algorithm ########
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf"))
def ula_kernel(x, key, step_size, grad_log_pdf, **kwargs):
    """
    Return the next state of the ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update state
    x = x + step_size * g + jnp.sqrt(2 * step_size) * jax.random.normal(
        subkey, x.shape)

    return (x, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin"))
def jax_ula_sampler(x0,
                    n_samples,
                    step_size,
                    grad_log_pdf,
                    burnin=None,
                    eps=None,
                    key=jax.random.PRNGKey(0),
                    **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def ula_step(carry, _):
        x, key = carry
        x, key = ula_kernel(x, key, step_size, grad_log_pdf, **kwargs)
        return (x, key), x

    carry = (x, key)
    _, samples = jax.lax.scan(ula_step, carry, None, length=n_samples)

    if burnin is not None:
        return samples[burnin:]

    return samples, None



############################## rmsprop ULA ##############################
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",
                                   ))
def rmsprop_ula_kernel(x, 
                       f, 
                       counter, 
                       key, 
                       step_size, 
                       beta, 
                       eps,
                       grad_log_pdf, 
                       **kwargs):
    """
    Return the next state of the rmsprop ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key, num=2)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update f
    f = beta * f + (1 - beta) * g**2

    # normalizing term
    c = jnp.sqrt(f) + eps
    
    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)
    
    counter += 1

    return (x, f, counter, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",
                          ))
def jax_rmsprop_ula_sampler(x0,
                            n_samples,
                            step_size,
                            beta,
                            eps,
                            grad_log_pdf,
                            burnin=None,
                            key=jax.random.PRNGKey(0),
                            return_Vs=False,
                            **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def rmsprop_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rmsprop_ula_kernel(x, 
                                                f, 
                                                counter, 
                                                key, 
                                                step_size, 
                                                beta,
                                                eps,
                                                grad_log_pdf, 
                                                **kwargs)
        # also return f
        return (x, f, counter, key), (x, f)

    # somehow it was initialised as ones instead of zeros
    carry = (x, 1.0 * jnp.zeros(x.shape), 1, key)
    
    _, out = jax.lax.scan(rmsprop_ula_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]

    # if return_Vs:
    #     return samples, Vs
    # else:
    #     return samples
    return samples, Vs

############################## rmsprop ULA ##############################
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",))
def rmsfull_ula_kernel(x, 
                       f, 
                       counter, 
                       key, 
                       step_size, 
                       beta, 
                       eps,
                       grad_log_pdf, 
                       **kwargs):
    """
    Return the next state of the rmsprop ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key, num=2)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update f
    f = beta * f + (1 - beta) * (jnp.matmul(g[:,jnp.newaxis], g[:, jnp.newaxis].T))
    
    # V = jax.scipy.linalg.sqrtm(f).astype(jnp.float32)
    V = f
    # update state
    # jax.debug.print("🤯 {x} 🤯", x=x)
    # jax.debug.print("🤯 {f} 🤯", f=f)
    
    x = x + step_size * (jax.scipy.linalg.inv(V + eps * jnp.diag(jnp.ones_like(x))) @ g) + jnp.sqrt(
        2 * step_size) * jax.scipy.linalg.sqrtm(\
            jax.scipy.linalg.inv(\
                V+ eps * jnp.diag(jnp.ones_like(x)))).\
            astype(jnp.float32)\
                @ jax.random.normal(subkey, x.shape)
    
    counter += 1

    return (x, f, counter, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps"))
def jax_rmsfull_ula_sampler(x0,
                            n_samples,
                            step_size,
                            beta,
                            eps,
                            grad_log_pdf,
                            burnin=None,
                            key=jax.random.PRNGKey(0),
                            return_Vs=False,
                            **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def rmsfull_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rmsfull_ula_kernel(x, 
                                                f, 
                                                counter, 
                                                key, 
                                                step_size, 
                                                beta,
                                                eps,
                                                grad_log_pdf, 
                                                **kwargs)
        # also return f
        return (x, f, counter, key), (x, f)

    carry = (x, 1.0 * jnp.zeros((x.shape[0], x.shape[0])), 1, key)
    
    _, out = jax.lax.scan(rmsfull_ula_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]
    
    # def return_samples(samples, Vs):
    #     return samples, None
    # def return_V(samples, Vs):
    #     return samples, Vs
    
    # return jax.lax.cond(return_Vs, 
    #              return_V, 
    #              return_samples, 
    #              samples, Vs)
    # if return_Vs:
    return samples, Vs
    # else:
    #     return samples


##### preconditioned ula with fixed preconditioner #####
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf",
                                   ))
def pula_kernel(x, 
                key, 
                step_size, 
                grad_log_pdf, 
                P,
                **kwargs):
    """
    Return the next state of the pULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update state
    x = x + step_size * jnp.matmul(P, g) + \
        jnp.sqrt(2 * step_size) * jnp.matmul(jax.scipy.linalg.sqrtm(P).astype(x.dtype),
                                             jax.random.normal(subkey, x.shape))

    return (x, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin"))
def jax_pula_sampler(x0,
                     n_samples,
                     step_size,
                     grad_log_pdf,
                     eps,
                     P,
                     burnin=None,
                     key=jax.random.PRNGKey(0),
                     **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def pula_step(carry, _):
        x, key = carry
        x, key = pula_kernel(x, key, step_size, grad_log_pdf, P, **kwargs)
        return (x, key), x

    carry = (x, key)
    _, samples = jax.lax.scan(pula_step, carry, None, length=n_samples)

    if burnin is not None:
        return samples[burnin:]

    return samples, None


############################## adamax without momentum ##############################
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",))
def rmax_ula_kernel(x, 
                    f, 
                    counter, 
                    key, 
                    step_size, 
                    beta, 
                    eps,
                    grad_log_pdf, 
                    **kwargs):
    """
    Return the next state of the rmax ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key, num=2)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update f
    f = jnp.maximum(beta * f, jnp.abs(g))

    # normalizing term
    c = f + eps
    
    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)
    
    counter += 1

    return (x, f, counter, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",))
def jax_rmax_ula_sampler(x0,
                        n_samples,
                        step_size,
                        beta,
                        eps,
                        grad_log_pdf,
                        burnin=None,
                        key=jax.random.PRNGKey(0),
                        return_Vs=False,
                        **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def rmax_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rmax_ula_kernel(x, 
                                                f, 
                                                counter, 
                                                key, 
                                                step_size, 
                                                beta,
                                                eps,
                                                grad_log_pdf, 
                                                **kwargs)
        # also return f
        return (x, f, counter, key), (x, f)

    carry = (x, 1.0 * jnp.ones(x.shape), 1, key)
    
    _, out = jax.lax.scan(rmax_ula_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]

    # if return_Vs:
    #     return samples, Vs
    # else:
    #     return samples
    
    return samples, Vs


############################## adahessian without momentum ##############################
@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",))
def adah_ula_kernel(x, 
                    f, 
                    counter, 
                    key, 
                    step_size, 
                    beta, 
                    eps,
                    grad_log_pdf, 
                    **kwargs):
    """
    Return the next state of the rmax ULA kernel.
    """

    # split key
    key, subkey1, subkey2 = jax.random.split(key, num=3)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # compute the Hessian diagonal using Hutchinson's estimator
    bern = jax.random.bernoulli(subkey2, 0.5, shape=x.shape)
    
    # Radmacher random variable
    rad = bern * 2 - 1
    
    # Hessian oracle
    Hz = jax.grad(lambda x: jnp.vdot(g, x))(rad)
    
    # Hessian diagonal
    H = rad * Hz

    # normalizing term
    # f = beta * f + (1 - beta) * H**2
    f = H
    c = f + eps
    
    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey1, x.shape)
    
    counter += 1

    return (x, f, counter, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",))
def jax_adah_ula_sampler(x0,
                        n_samples,
                        step_size,
                        beta,
                        eps,
                        grad_log_pdf,
                        burnin=None,
                        key=jax.random.PRNGKey(0),
                        return_Vs=False,
                        **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def adah_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rmax_ula_kernel(x, 
                                            f, 
                                            counter, 
                                            key, 
                                            step_size, 
                                            beta,
                                            eps,
                                            grad_log_pdf, 
                                            **kwargs)
        # also return f
        return (x, f, counter, key), (x, f)

    carry = (x, 1.0 * jnp.ones(x.shape), 1, key)
    
    _, out = jax.lax.scan(adah_ula_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]

    # if return_Vs:
    #     return samples, Vs
    # else:
    #     return samples
    return samples, Vs

############################################################
samplers_dict = {
    "ula": jax_ula_sampler,
    "rmsprop": jax_rmsprop_ula_sampler,
    "rmsfull": jax_rmsfull_ula_sampler,
    "pula": jax_pula_sampler,
    "rmax": jax_rmax_ula_sampler,
    "adahessian": jax_adah_ula_sampler,
}


def get_samplers(name, fix=False, hyperparam={}):
    """
    Return a partial function of the sampler with fixed hyperparameters.  
    
    Args:  
        - name: name of the sampler.
        - fix: whether to fix the hyperparameters.
        - hyperparam: hyperparameters to fix, dictionary.
    """
    if not fix:
        return samplers_dict[name]
    else:
        # fix the hyperparameters
        for key in hyperparam.keys():
            if key not in inspect.getfullargspec(samplers_dict[name]).args:
                # raise ValueError("Hyperparameter {} not found in {}.".format(key, 
                                                                            #  samplers_dict[name].__name__))
                return samplers_dict[name]
        return partial(samplers_dict[name], **hyperparam)