import jax
import scipy
import pints.toy
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from functools import partial



def gaussian_pdf(x: jnp.ndarray, mu: jnp.ndarray,
                 sigma: jnp.ndarray) -> jnp.ndarray:
    ret = jax.scipy.stats.multivariate_normal.pdf(x, mu, sigma)
    return ret


def grad_gaussian_logpdf(x: jnp.ndarray, mu: jnp.ndarray,
                         sigma: jnp.ndarray) -> jnp.ndarray:
    return -jnp.linalg.solve(sigma, np.identity(2)) @ (x - mu)

def fisher_gaussian(x: jnp.ndarray, mu: jnp.ndarray,
                    sigma: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(sigma, jnp.identity(2))


def banana_pdf(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.1 * (v[0]**2) - 0.1 * (v[1]**4) - 2 *
                   ((v[1] - v[0]**2)**2))


def grad_banana_logpdf(v: jnp.ndarray) -> jnp.ndarray:
    x = v[0]
    y = v[1]
    return jnp.array(
        [-x / 5 + 8 * x * (y - x**2), (-0.4) * (y**3) - 4 * (y - x**2)])
    
    
def mix_gaussian_pdf2(x,
                      z1,
                      z2,
                      mu1,
                      mu2,
                      sigma1=jnp.identity(2),
                      sigma2=jnp.identity(2)):
    """Mixture of two Gaussians."""
    return z1 * gaussian_pdf(x, mu1, sigma1) \
    + z2 * gaussian_pdf(x, mu2, sigma2)


def grad_mix_gaussian_logpdf2(x,
                              z1,
                              z2,
                              mu1,
                              mu2,
                              sigma1=jnp.identity(2),
                              sigma2=jnp.identity(2)):
    
    """Gradient of log mixture of two Gaussians."""
    f1 = gaussian_pdf(x, mu1, sigma1)
    f2 = gaussian_pdf(x, mu2, sigma2)

    return 1/(1 + z2 * f2 / (z1 * f1)) * grad_gaussian_logpdf(x, mu1, sigma1) \
            + 1/(1 + z1 * f1 / (z2 * f2)) * grad_gaussian_logpdf(x, mu2, sigma2)
            
            
def mix_gaussian_pdf5(x, mu_l):
    ret = 0
    for mu in mu_l:
        ret += gaussian_pdf(x, mu, jnp.identity(2))
    return ret / 5


def grad_mix_gaussian_logpdf5(x, mu_l):
    ret = 0
    for mu in mu_l:
        ret += gaussian_pdf(x, mu, jnp.identity(2))*\
            (-2 * (x - mu)) / 5
    return ret / mix_gaussian_pdf5(x, mu_l)


def funnel_pdf(nu, data, dimension=2):
    """Neal's funnel distribution"""
    # assert data.shape[0] == dimension
    logpdf = jax.scipy.stats.norm.logpdf(nu, 0, 3)
    for i in range(dimension):
        # all standard deviation not variance
        logpdf += jax.scipy.stats.norm.logpdf(data[i], 0, jnp.exp(nu / 2))
        
    return jnp.exp(logpdf)


def grad_funnel_logpdf_auto(nu, data):
    dimension = data.shape[0]
    func = partial(funnel_pdf, data=data, dimension=dimension)
    def logpdf(nu):
        return jnp.log(func(nu))
    return jax.grad(logpdf)(nu)

def grad_funnel_logpdf(nu, data):
    dimension = data.shape[0]
    data_grad = jnp.sum(data**2) * jnp.exp(-nu) / 2. -0.5 * dimension
    prior_grad = - nu/9.
    return data_grad + prior_grad

def funnel_normalizing_constant(data, interval=[-10., 10.]):
    """Numerically integrate the funnel distribution"""
    dimension = data.shape[0]
    func = partial(funnel_pdf, data=data, dimension=dimension)
    return scipy.integrate.quad(func, a=interval[0], b=interval[1])[0]





