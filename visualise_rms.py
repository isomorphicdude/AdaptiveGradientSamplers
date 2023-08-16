import argparse
import numpy as np
import jax
import jax.numpy as jnp

import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial


argparser = argparse.ArgumentParser()

argparser.add_argument('--seed', type=int, default=1024)
argparser.add_argument('--lr', type=float, default=2)
argparser.add_argument('--density', type=str, default='mix_gaussian2')
argparser.add_argument('--N', type=int, default=10000)
argparser.add_argument('--beta1', type=float, default=0.4)
argparser.add_argument('--beta2', type=float, default=0.99)
argparser.add_argument('--scale', type=float, default=100)
argparser.add_argument('--shift', type=float, default=800000)
argparser.add_argument('--anneal', type=bool, default=False)

# number of samples
N = argparser.parse_args().N

# density name
density = argparser.parse_args().density

# beta parameters for interpolation
beta1 = argparser.parse_args().beta1
beta2 = argparser.parse_args().beta2

# scale and shift parameters for sigmoid interpolation
scale = argparser.parse_args().scale
shift = argparser.parse_args().shift

# learning rate and annealing and random seed
seed = argparser.parse_args().seed
lr = argparser.parse_args().lr
anneal = argparser.parse_args().anneal

key = jax.random.PRNGKey(seed)

################## GAUSSIAN ##################
# mean
mu = jnp.array([0., 0.])

# covariance matrix
sigma = jnp.array([[1e-2, -0.02], [-0.02, 1.0]])

def gaussian_pdf(x: jnp.ndarray, mu: jnp.ndarray,
                 sigma: jnp.ndarray) -> jnp.ndarray:
    ret = jax.scipy.stats.multivariate_normal.pdf(x, mu, sigma)
    return ret


def grad_gaussian_logpdf(x: jnp.ndarray, mu: jnp.ndarray,
                         sigma: jnp.ndarray) -> jnp.ndarray:
    return -jnp.linalg.solve(sigma, np.identity(2)) @ (x - mu)



################## BANANA ##################
def banana_pdf(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.1 * (v[0]**2) - 0.1 * (v[1]**4) - 2 *
                   ((v[1] - v[0]**2)**2))


def grad_banana_logpdf(v: jnp.ndarray) -> jnp.ndarray:
    x = v[0]
    y = v[1]
    return jnp.array(
        [-x / 5 + 8 * x * (y - x**2), (-0.4) * (y**3) - 4 * (y - x**2)])



################## 2 MIXTURE OF GAUSSIAN ##################
# mean
mu1 = jnp.array([-5.0, -5.0])
mu2 = np.array([5.0, 5.0])

# coefficients
z1 = 0.2
z2 = 0.8

# covariance matrices
sigma1 = np.identity(2)
sigma2 = np.identity(2)

def mix_gaussian_pdf2(x,
                      z1,
                      z2,
                      mu1,
                      mu2,
                      sigma1=jnp.identity(2),
                      sigma2=jnp.identity(2)):
    return z1 * gaussian_pdf(x, mu1, sigma1) \
    + z2 * gaussian_pdf(x, mu2, sigma2)


def grad_mix_gaussian_logpdf2(x,
                              z1,
                              z2,
                              mu1,
                              mu2,
                              sigma1=jnp.identity(2),
                              sigma2=jnp.identity(2)):
    f1 = gaussian_pdf(x, mu1, sigma1)
    f2 = gaussian_pdf(x, mu2, sigma2)

    return 1/(1 + z2 * f2 / (z1 * f1)) * grad_gaussian_logpdf(x, mu1, sigma1) \
            + 1/(1 + z1 * f1 / (z2 * f2)) * grad_gaussian_logpdf(x, mu2, sigma2)


################## 5 MIXTURE OF GAUSSIAN ##################
mu_l = [
    jnp.array([-3, -3]),
    jnp.array([-3, 3]),
    jnp.array([0, 0]),
    jnp.array([3, -3]),
    jnp.array([3, 3])
]

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



######################## IMPLEMENTATION OF RMSprop-ULA ########################
@partial(jax.jit, static_argnames=("step_size", 
                                   "N",
                                   "grad_log_pdf", 
                                    "beta1",
                                   "beta2", 
                                   "scale",
                                   "shift",
                                   "anneal"))
def rmsprop_ula_kernel(x, f, 
                       N,
                       counter, 
                       key, 
                       step_size, 
                       beta1, 
                       beta2,
                       scale,
                       shift,
                       grad_log_pdf, 
                       anneal, **kwargs):
    """
    Return the next state of the ULA kernel.
    """

    # split key
    key, subkey = jax.random.split(key)

    # gradient of log pdf
    g = grad_log_pdf(x, **kwargs)
    
    # interpolation by Sigmoid
    if anneal:
        # A sigmoid schedule for beta
        beta = beta1 * (1 - (1+jnp.exp(-scale*(counter-shift)/N))**(-1)) + \
            beta2 * (1+jnp.exp(-scale*(counter-shift)/N))**(-1)
        
        # A linear schedule for beta
        # beta = beta1 * (1 - ((counter-shift)/N)/scale) + beta2 * ((counter-shift)/N) / scale
        
        # A sinusoidal schedule for beta
        # beta = jnp.abs(jnp.sin(counter/(N*10)*jnp.pi/2)) * beta2
    else:
        beta = beta2

    # update f
    f = beta * f + (1 - beta) * g**2

    # normalizing term
    c = jnp.sqrt(f + 1e-8)
    
    # update state
    x = x + step_size * (g / c) + jnp.sqrt(
        2 * step_size / c) * jax.random.normal(subkey, x.shape)
    
    counter += 1

    return (x, f, counter, key)


@partial(jax.jit,
         static_argnames=("n_samples", "step_size", "grad_log_pdf", "burnin",
                          "beta1", "beta2", "scale", "shift",
                          "anneal"))
def rmsprop_ula_sampler(x0,
                        n_samples,
                        step_size,
                        beta1,
                        beta2,
                        scale,
                        shift,
                        grad_log_pdf,
                        anneal=False,
                        burnin=None,
                        key=jax.random.PRNGKey(0),
                        **kwargs):
    """
    Return samples from unadjusted Langevin algorithm.  
    """
    x = x0

    # use for scan
    def rmsprop_ula_step(carry, _):
        x, f, counter, key = carry
        x, f, counter, key = rmsprop_ula_kernel(x, f, n_samples, counter, key, 
                                                step_size, 
                                                beta1, beta2,
                                                scale, shift,
                                       grad_log_pdf, anneal, **kwargs)
        return (x, f, counter, key), x

    carry = (x, 1.0 * jnp.ones(x.shape), 1, key)
    
    _, samples = jax.lax.scan(rmsprop_ula_step, carry, None, length=n_samples)

    if burnin is not None:
        return samples[burnin:]

    return samples


############## MAIN ##############
def main():
    # samples from rmsprop ULA
    if density=="gaussian":
        samples = rmsprop_ula_sampler(jnp.array([10., 10.]),
                                                N * 10,
                                                lr,
                                                beta1,
                                                beta2,
                                                scale,
                                                shift,
                                                grad_gaussian_logpdf,
                                                burnin=0,
                                                mu=mu,
                                                sigma=sigma,
                                                anneal=anneal,
                                                key = key)
    elif density=="banana":
        samples = rmsprop_ula_sampler(jnp.array([4., 4.]),
                                          N * 10,
                                          lr,
                                          beta1,
                                          beta2,
                                          scale,
                                          shift,
                                          grad_banana_logpdf,
                                          burnin=0,
                                          anneal=anneal,
                                          key = key)
    elif density=="mix_gaussian2":
        # samples from rmsprop ULA
        samples = rmsprop_ula_sampler(jnp.array([0., 0.]),
                                                N * 10,
                                                lr,
                                                beta1,
                                                beta2,
                                                scale,
                                                shift,
                                                grad_mix_gaussian_logpdf2,
                                                burnin=0,
                                                anneal=anneal,
                                                z1=z1,
                                                z2=z2,
                                                mu1=mu1,
                                                mu2=mu2,
                                                key = key)
    elif density=='mix_gaussian5':
        # samples from rmsprop ULA
        samples = rmsprop_ula_sampler(jnp.array([10.0, 10.0]),
                                                N * 10,
                                                lr,
                                                beta1,
                                                beta2,
                                                scale,
                                                shift,
                                                grad_mix_gaussian_logpdf5,
                                                anneal=anneal,
                                                burnin=None,
                                                mu_l=mu_l,
                                                key = key)
        
        
    # plot
    plt.figure(figsize=(5, 5))
    for i in range(len(samples)):
        every = N * 10 // 20
        if i % every == 0:
            plt.clf()
            if anneal:
                beta = beta1 * (1 - (1+jnp.exp(-scale*(i-shift)/(N*10)))**(-1)) + \
                beta2 * (1+jnp.exp(-scale*(i-shift)/(N*10)))**(-1)
                
                # beta = beta1 * (1 - ((i-shift)/(N*10))/scale) + beta2 * ((i-shift)/(N*10)) / scale
                # beta = jnp.sin(i/(N*10)*jnp.pi/2)
            else:
                beta = beta2
            plt.hist2d(samples[:i, 0], samples[:i, 1], bins=100, cmap='RdBu')
            
            plt.title(f"RMSprop-ULA, step {i}, beta={beta}")
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.pause(0.5)
            
    # plot final
    plt.clf()
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap='RdBu')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)    
    plt.title(f"RMSprop-ULA, step {N*10}, beta={beta}")
            
    import time
    time.sleep(10)
            
if __name__ == "__main__":
    print("You have chosen density: ", density)
    print("You have chosen beta1: ", beta1)
    print("You have chosen beta2: ", beta2)
    print("You have chosen lr: ", lr)
    print("You have chosen seed: ", seed)
    print("You have chosen anneal to be: ", anneal)
    print("You have chosen scale to be: ", scale)
    print("You have chosen shift to be: ", shift)
    main()
    