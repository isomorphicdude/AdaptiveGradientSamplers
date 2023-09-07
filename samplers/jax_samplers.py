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
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.  
        
    Returns:  
        - samples: generated samples.
        - None: dummy return.
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
    
    # step_size = step_size * jnp.linalg.norm(c)
    
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
    Return samples from rmsprop ula.   
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of squared gradient.
        - eps: control extreme values of the Vs
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.
        - return_Vs: whether to return the Vs  
        
    Returns:  
        - samples: generated samples.
        - Vs: Vs list.
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

############################## rmsfull ULA ##############################
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
    Return samples from rmsfull ula.  
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of outer product.
        - eps: control extreme values of the Vs  
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.
        - return_Vs: whether to return the Vs  
        
    Returns:  
        - samples: generated samples.
        - Vs: Vs list.
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

    carry = (x, jnp.zeros((x.shape[0], x.shape[0])), 1, key)
    
    _, out = jax.lax.scan(rmsfull_ula_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]
    
    return samples, Vs


########################################## Adam ULA ##########################################

@partial(jax.jit,
         static_argnames=("step_size", 
                          "grad_log_pdf", 
                          "beta1", 
                          "beta2",
                          "eps"))
def adam_ula_kernel(x, 
                    m, 
                    s, 
                    iter, 
                    key, 
                    step_size, 
                    beta1, 
                    beta2, 
                    eps,
                    grad_log_pdf,
                    **kwargs):
    """
    Return the next state of the ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # update m and s
    m = beta1 * m + (1 - beta1) * g
    s = beta2 * s + (1 - beta2) * (g**2)

    # bias correction
    m_hat = m / (1 - beta1**(iter + 1))
    s_hat = s / (1 - beta2**(iter + 1))

    # normalizing term
    c = jnp.sqrt(s_hat + eps)

    # update state
    x = x + step_size * (m_hat / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)

    iter += 1

    return (x, m, s, iter, key)


@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta1", 
                          "beta2",
                          "eps"))
def jax_adam_ula_sampler(x0,
                     n_samples,
                     step_size,
                     beta1,
                     beta2,
                     eps,
                     grad_log_pdf,
                     burnin=None,
                     key=jax.random.PRNGKey(0),
                     **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """

    # use for scan
    def adam_ula_step(carry, _):
        x, m, s, iter, key = carry
        x, m, s, iter, key = adam_ula_kernel(x, m, s, iter, key, step_size,
                                             beta1, beta2, eps, grad_log_pdf,
                                             **kwargs)

        return (x, m, s, iter, key), (x, (m, s))

    carry = (x0, jnp.zeros(x0.shape), jnp.zeros(x0.shape), 0, key)
    _, out = jax.lax.scan(adam_ula_step, carry, None, length=n_samples)

    samples, other = out
    
    if burnin is not None:
        return samples[burnin:]

    return samples, other


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
                          "burnin",
                          "isotropic"))
def jax_pula_sampler(x0,
                     n_samples,
                     step_size,
                     grad_log_pdf,
                     eps,
                     P,
                     burnin=None,
                     key=jax.random.PRNGKey(0),
                     isotropic=False,
                     **kwargs):
    """
    Return samples from preconditioned unadjusted Langevin algorithm.  
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.
        - isotropic: whether to use isotropic preconditioner.
        
    Returns:  
        - samples: generated samples.
        - None: dummy return.
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


############################## rmax without momentum ##############################
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
    Return samples from rmsprop ula with l-infinity norm.
    Follows from the ADAMAX from Kingma and Ba (2014).
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of max gradient.
        - eps: control extreme values of the Vs
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.
        - return_Vs: whether to return the Vs  
        
    Returns:  
        - samples: generated samples.
        - Vs: Vs list.
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
    rad = bern.astype(jnp.float32) * 2. - 1.
    
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
    Return samples from adahessian ula.
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of diagonal Hessian.
        - eps: control extreme values of the diagonal Hessian.
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.
        - return_Vs: whether to return the diagonal Hessian.
        
    Returns:  
        - samples: generated samples.
        - Vs: diagonal Hessian list.
    """
    x = x0

    # use for scan
    def adah_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = adah_ula_kernel(x, 
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


############################## Monge (2023 Yu et al.) ##############################
@partial(jax.jit, static_argnames=("dim",
                                   "alpha_2"))
def _get_monge_metrics(dim, grad, alpha_2):
    grad_norm_2 = jnp.linalg.norm(grad)**2
    
    # forming the inverse G^-1
    G_r = jnp.eye(dim) - alpha_2 / (1 + alpha_2 * grad_norm_2) *\
        (grad[:,jnp.newaxis] @ grad[:,jnp.newaxis].T)
    
    # forming the inverse sqrt G^-1/2
    G_rsqrt = jnp.eye(dim) +\
        (1 / grad_norm_2) * (1 / jnp.sqrt(1 + alpha_2 * grad_norm_2) - 1) *\
            (grad[:,jnp.newaxis] @ grad[:,jnp.newaxis].T)
            
    return G_r, G_rsqrt

@partial(jax.jit, static_argnames=("step_size",))
def _get_step_big(g, 
                  x,
                  precond_grad, 
                  threshold, 
                  step_size, 
                  subkey, 
                  G_rsqrt):
    
    factor = jnp.linalg.norm(g) / threshold
    
    update_step = g / factor * step_size +\
        jax.random.normal(subkey, x.shape) / jnp.sqrt(factor) * jnp.sqrt(2 * step_size)
        
    return update_step

@partial(jax.jit, static_argnames=("step_size",))
def _get_step_normal(g, 
                     x,
                     precond_grad,
                     threshold, 
                     step_size, 
                     subkey, 
                     G_rsqrt):
    update_step = precond_grad * step_size +\
        G_rsqrt @ jax.random.normal(subkey, x.shape) * jnp.sqrt(2 * step_size)
        
    return update_step

@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "alpha_2",
                                   "lambd",
                                   "eps",))
def monge_ula_kernel(x, 
                     p_grad, 
                     counter, 
                     key, 
                     step_size, 
                     alpha_2, 
                     lambd,
                     eps,
                     grad_log_pdf, 
                     **kwargs):
    """
    Return the next state of the Monge ULA kernel.
    """
    dim = x.shape[0]
    threshold = 1e3
    
    # split key
    key, subkey = jax.random.split(key, num=2)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    # moving average of the gradient
    p_grad = lambd * g + (1 - lambd) * p_grad # usually noisy gradient
    p_grad = g
    
    # get preconditioners
    G_r, G_rsqrt = _get_monge_metrics(dim, p_grad, alpha_2)
    
    # precond grad
    precond_grad = G_r @ g
    
    x_shape = x.shape
    update_step = jax.lax.cond(jnp.linalg.norm(precond_grad) > threshold,
                               _get_step_big,
                               _get_step_normal,
                               g,
                               x,
                               precond_grad,
                               threshold,
                               step_size,
                               subkey,
                               G_rsqrt)                             
            
    # update state
    x = x + update_step
    counter += 1
    
    return (x, p_grad, counter, key)


@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",))
def jax_monge_ula_sampler(x0,
                          n_samples,
                          step_size,
                          alpha_2,
                          lambd,
                          eps,
                          grad_log_pdf,
                          burnin=None,
                          key=jax.random.PRNGKey(0),
                          return_Vs=False,
                          **kwargs):
    """
    Return samples from monge SGLD.  
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - alpha_2: alpha^2 in the paper.
        - lambd: lambda for moving average of gradient.
        - eps: dummy constant not used
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - key: random key.  
        
    Returns:  
        - samples: generated samples.
        - None: dummy return.
    """
    x = x0

    # use for scan
    def monge_ula_step(carry, _):
        x, p, counter, key = carry
        x, p, counter, key = monge_ula_kernel(x,
                                              p,
                                              counter,
                                              key,
                                              step_size,
                                              alpha_2,
                                              lambd,
                                              None,
                                              grad_log_pdf,
                                              **kwargs)
        
        return (x, p, counter, key), x

    carry = (x, jnp.zeros(x.shape), 1, key)
    
    _, samples = jax.lax.scan(monge_ula_step, carry, None, length=n_samples)
    
    if burnin is not None:
        return samples[burnin:]
    
    return samples, None


############################### Forgetful RMSProp ###############################

# @partial(jax.jit, static_argnames=("f",))
def _forget_f(f):
    return jnp.zeros(f.shape)

# @partial(jax.jit, static_argnames=("f",))
def _remember_f(f):
    return f

@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",
                                #    "forget_times",
                                   ))
def rms_forgetf_kernel(x, 
                       f, 
                       counter, 
                       key, 
                       step_size, 
                       beta, 
                       eps,
                       grad_log_pdf, 
                       forget_times,
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
    
    # step_size = step_size * jnp.linalg.norm(c)
    
    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)
    
    f = jax.lax.cond(jnp.isin(counter, forget_times),
                     _forget_f,
                     _remember_f,
                     f)

    counter += 1
    
    return (x, f, counter, key)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",
                        #   "forget_times",
                          ))
def jax_rms_forgetf_sampler(x0,
                            n_samples,
                            step_size,
                            beta,
                            eps,
                            grad_log_pdf,
                            burnin=None,
                            forget_times = None,
                            key=jax.random.PRNGKey(0),
                            return_Vs=False,
                            **kwargs):
    """
    Return samples from rmsprop ula.   
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of squared gradient.
        - eps: control extreme values of the Vs
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - forget_times: the times the V_t is refreshed
        - key: random key.
        - return_Vs: whether to return the Vs  
        
    Returns:  
        - samples: generated samples.
        - Vs: Vs list.
    """
    x = x0

    # use for scan
    def rms_forgetf_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rms_forgetf_kernel(x, 
                                                f, 
                                                counter, 
                                                key, 
                                                step_size, 
                                                beta,
                                                eps,
                                                grad_log_pdf, 
                                                forget_times,
                                                **kwargs)
        # also return f
        return (x, f, counter, key), (x, f)

    # somehow it was initialised as ones instead of zeros
    carry = (x, 1.0 * jnp.zeros(x.shape), 1, key)
    
    _, out = jax.lax.scan(rms_forgetf_step, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]

    # if return_Vs:
    #     return samples, Vs
    # else:
    #     return samples
    return samples, Vs


####################################### forgetful rmsprop 2 #######################################

#NOTE: this does not refresh the gradients, but stores only the number of gradients according to 
# the window size, then usual weighting is applied, the stored gradients form a queue

@partial(jax.jit, static_argnames=("step_size", 
                                   "beta",
                                   "eps",
                                   "window_size",
                                   "seq_exp",
                                   ))
def _update_before(x, 
                   f, 
                   counter,
                   g, 
                   subkey, 
                   step_size, 
                   beta, 
                   eps,
                   window_size,
                   past_grads,
                   seq_exp):
    # update f
    f = beta * f + (1 - beta) * g**2

    # normalizing term
    c = jnp.sqrt(f) + eps

    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)

    # update past_grads
    past_grads = past_grads.at[counter].set(g)

    return (x, f, past_grads)

@partial(jax.jit, static_argnames=("step_size",
                                      "beta",
                                      "eps",
                                      "window_size",
                                      "seq_exp",
                                      ))
def _update_after(x, 
                  f, 
                  counter,
                  g,
                  subkey, 
                  step_size, 
                  beta, 
                  eps,
                  window_size,
                  past_grads,
                  seq_exp):
    
    # update past_grads
    _prev_grads = past_grads[1:]
    past_grads = jnp.concatenate((_prev_grads, g[jnp.newaxis, :]), axis=0)
    
    # update f
    f = 0.0

    f = jnp.sum(
        (1-beta) * (past_grads**2) * (beta**(seq_exp)), axis=0
    )
    # normalizing term
    c = jnp.sqrt(f) + eps

    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)

    # print(f"After x {x} 🤯,  f {f}, past_grads {past_grads} 🤯 {len(past_grads)}")
    return (x, f, past_grads)
    
    

@partial(jax.jit, static_argnames=("step_size", 
                                   "grad_log_pdf", 
                                   "beta",
                                   "eps",
                                   "seq_exp",
                                   ))
def rms_forgetf_kernel2(x, 
                       f, 
                       counter, 
                       key, 
                       step_size, 
                       beta, 
                       eps,
                       grad_log_pdf, 
                       past_grads,
                       window_size,
                       seq_exp,
                       **kwargs):
    """
    Return the next state of the rmsprop ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key, num=2)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)

    x, f, past_grads = jax.lax.cond(counter < window_size + 1,
                                    _update_before,
                                    _update_after,
                                    x,
                                    f,
                                    counter,
                                    g,
                                    subkey,
                                    step_size,
                                    beta,
                                    eps,
                                    window_size,
                                    past_grads,
                                    seq_exp)

    counter += 1
    
    return (x, f, counter, key, past_grads)

@partial(jax.jit,
         static_argnames=("n_samples", 
                          "step_size", 
                          "grad_log_pdf", 
                          "burnin",
                          "beta", 
                          "eps",
                          "window_size",
                          ))
def jax_rms_forgetf_sampler2(x0,
                            n_samples,
                            step_size,
                            beta,
                            eps,
                            grad_log_pdf,
                            burnin=None,
                            window_size = 50,
                            key=jax.random.PRNGKey(0),
                            return_Vs=False,
                            **kwargs):
    """
    Return samples from rmsprop ula.   
    
    Args:  
        - x0: initial state.
        - n_samples: number of samples to generate.
        - step_size: step size.
        - beta: beta for moving average of squared gradient.
        - eps: control extreme values of the Vs
        - grad_log_pdf: gradient of log pdf.
        - burnin: number of burnin samples.
        - window_size: the number of gradients to store
        - key: random key.
        - return_Vs: whether to return the Vs  
        
    Returns:  
        - samples: generated samples.
        - Vs: Vs list.
    """
    x = x0
    seq_exp = (-window_size + jnp.arange(window_size))[:, jnp.newaxis]
    
    def rms_forgetf_step2(carry, _):
        x, f, counter, key, past_grads = carry
        x, f, counter, key, past_grads = rms_forgetf_kernel2(x, 
                                                f, 
                                                counter, 
                                                key, 
                                                step_size, 
                                                beta,
                                                eps,
                                                grad_log_pdf, 
                                                past_grads,
                                                window_size,
                                                seq_exp,
                                                **kwargs)
        # also return f
        return (x, f, counter, key, past_grads), (x, f)
    
    # use a [] for queue
    carry = (x, jnp.zeros(x.shape), 0, key, jnp.zeros((window_size, x.shape[0])))
    
    _, out = jax.lax.scan(rms_forgetf_step2, carry, None
                              , length=n_samples)

    samples, Vs = out
    
    if burnin is not None:
        return samples[burnin:]

    return samples, Vs






############################################################
samplers_dict = {
    "ula": jax_ula_sampler,
    "rmsprop": jax_rmsprop_ula_sampler,
    "rmsfull": jax_rmsfull_ula_sampler,
    "pula": jax_pula_sampler,
    "rmax": jax_rmax_ula_sampler,
    "adam": jax_adam_ula_sampler,
    "adahessian": jax_adah_ula_sampler,
    "monge": jax_monge_ula_sampler,
    "forgetful1": jax_rms_forgetf_sampler,
    "forgetful2": jax_rms_forgetf_sampler2,
}

def check_sampler_name(method_list):
    """
    Check if the sampler name is valid.  
    
    Args:  
        - method_list: list of samplers to check.
    """
    for method in method_list:
        if method not in samplers_dict.keys():
            raise ValueError("Sampler {} not found.".format(method))


def get_samplers(name, fix=False, hyperparam={}):
    """
    Return a partial function of the sampler with fixed hyperparameters.  
    
    Args:  
        - name: name of the sampler, one of the following:
            - ula
            - rmsprop
            - rmsfull
            - pula
            - rmax
            - adahessian
            - monge  
            - forgetful_rmsprop
            - forgetful_rmsprop2
            - forgetful_rmsfull
        - fix: whether to fix the hyperparameters.
        - hyperparam: hyperparameters to fix, dictionary.
    """
    if not fix:
        return samplers_dict[name]
    else:
        # fix the hyperparameters
        # print("Fixing hyperparameters for {}.".format(name))
        for key in hyperparam.keys():
            if key not in inspect.getfullargspec(samplers_dict[name]).args:
                # raise ValueError("Hyperparameter {} not found in {}.".format(key, 
                                                                            #  samplers_dict[name].__name__))
                return samplers_dict[name]
        return partial(samplers_dict[name], **hyperparam)