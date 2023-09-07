import abc
import jax
import jax.numpy as jnp
import numpy as np
import pints.toy
from functools import partial
from diagnostics import *

# Base class
class toy(abc.ABC):
    
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def pdf(x):
        pass
    
    @abc.abstractmethod
    def logpdf(x):
        pass
    
    @abc.abstractmethod
    def grad_logpdf(x):
        pass
    
    @abc.abstractmethod
    def kl_divergence(self, samples):
        pass
    
    @abc.abstractmethod
    def plot(self, ax, xlim, ylim, step, **kwargs):
        """
        Plot the density.(no samples)
        
        Args:  
            - ax: matplotlib axis
            - xlim: x-axis limits
            - ylim: y-axis limits
            - step: step size for the grid
            - kwargs: additional arguments for the density plot
        """
        xy = np.mgrid[-xlim:xlim:step, -ylim:ylim:step]
        vec_x = xy[np.newaxis, ...].reshape(2, -1).T
        out = ax.contourf(xy[0], 
                          xy[1], 
                          self.pdf(vec_x).reshape(int(2*xlim/step),
                                                    int(2*ylim/step)),
                          cmap="RdBu",
                          **kwargs)
        return out


class Gaussian2D(toy):
    """2D Gaussian toy model."""
    
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        assert sigma.shape == (2, 2)
        assert mu.shape == (2,)
        assert np.all(np.linalg.eigvals(sigma) > 0)
        
    def ground_truth_samples(self, n_samples):
        """Generate samples from the ground truth normal distribution."""
        return jax.scipy.stats.\
            multivariate_normal.rvs(mean=self.mu, cov=self.sigma, size=n_samples)
    
    def pdf(self, x):
        return jax.scipy.stats.multivariate_normal.pdf(x, self.mu, self.sigma)
    
    def logpdf(self, x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma)
    
    def grad_logpdf(self, x):
        return -jnp.linalg.solve(self.sigma, np.identity(2)) @ (x - self.mu)
    
    def kl_divergence(self, samples):
        return pints.toy.GaussianLogPDF(mean=self.mu, 
                                        sigma=self.sigma).kl_divergence(samples)
        
    def wass2d(self, samples):
        mu0 = jnp.mean(samples, axis=0)
        mu1 = self.mu
        cov0 = jnp.cov(samples.T)
        cov1 = self.sigma
        
        w2d = 0.5 * (jnp.linalg.norm(mu0 - mu1)**2 + \
                jnp.trace(cov0 + cov1 - 2 * jax.scipy.linalg.sqrtm(cov0 @ cov1).\
                    astype(jnp.float32)))
        
        return w2d
    
    def plot(self, ax, xlim, ylim, step=0.5):
        super().plot(ax, xlim, ylim, step)


class Banana(toy):
    """The Banana distribution (upwards facing U-shaped)."""
    def __init__(self):
        super().__init__()
        
    def pdf(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.1 * (v[0]**2) - 0.1 * (v[1]**4) - 2 *
                    ((v[1] - v[0]**2)**2))
        
    def logpdf(v: jnp.ndarray) -> jnp.ndarray:
        return -0.1 * (v[0]**2) - 0.1 * (v[1]**4) - 2 * ((v[1] - v[0]**2)**2)
        
    def grad_logpdf(v: jnp.ndarray) -> jnp.ndarray:
        x = v[0]
        y = v[1]
        return jnp.array(
            [-x / 5 + 8 * x * (y - x**2), (-0.4) * (y**3) - 4 * (y - x**2)])
        
    def kl_divergence(self, samples):
        raise NotImplementedError(f"For simple Banana the divergence is not implemented.")
    
    def plot(self, ax, xlim, ylim, step):
        super().plot(ax, xlim, ylim, step)
    

class TwistedGaussian2D(toy):
    """
    Twisted Gaussian 2D from https://doi.org/10.1007/s001800050022
    
    - b: parameter controlling the degree of twisting
    - V: variance of first component, default 100
    - dim: dimension of the distribution, default 2
    """
    def __init__(self, b, V=100):
        super().__init__()
        self.b = b
        self.V = V
                
        # create mean and variance
        self._mean = jnp.zeros(2)
        
        self._cov = jnp.array([[100, 0],
                                 [0, 1]])
        
        # pdf
        self._pdf = lambda x, y: 1 / jnp.sqrt(2 * jnp.pi) * \
            jnp.exp(-0.5 * (1 / self.V * x**2 + y**2)) * \
                (1 / jnp.sqrt(self.V))
        
        # logpdf
        self._logpdf = lambda x, y: -0.5 * (jnp.log(2 * jnp.pi)) - \
            0.5 * (1 / self.V * x**2 + y**2) - 0.5 * jnp.log(self.V)
        
        
    def pdf(self, x, y):
        """Evaluate the pdf at (x,y)"""
        return self._pdf(x,
                        y+self.b*x**2-self.b*self.V)
        
    def logpdf(self, x, y):
        """Evaluate the logpdf at (x,y)."""
        return self._logpdf(x,
                            y+self.b*x**2-self.b*self.V)
        
    
    def grad_logpdf(self, z, autograd=True):
        """Evaluate the gradient of the logpdf at (x,y)."""
        x = z[0]
        y = z[1]
        if autograd:
            return jnp.array([jax.grad(self.logpdf, 0)(x, y),
                                jax.grad(self.logpdf, 1)(x, y)])
        else:
            return jnp.array([
                - x / self.V - (y + self.b * x**2 - self.b * self.V) * 2 * self.b * x,
                - y - self.b * x**2 + self.b * self.V
            ])
        
    
    def _untwist(self, x, y):
        """Untwist the samples; transform to Gaussian."""
        return x, y + self.b * x**2 - self.b * self.V
        
    
    def kl_divergence(self, samples):
        """Computes KL between normal distributions"""
        untwistedx, untwistedy = self._untwist(samples[:, 0], samples[:, 1])
        samples = jnp.array([untwistedx, untwistedy]).T
        return pints.toy.GaussianLogPDF(mean=self._mean,
                                        sigma=self._cov).kl_divergence(samples)
        
        
    def wass2d(self, samples):
        """Computes 2D Wasserstein distance between samples and the ground truth."""
        untwistedx, untwistedy = self._untwist(samples[:, 0], samples[:, 1])
        samples = jnp.array([untwistedx, untwistedy]).T
        
        mu0 = jnp.mean(samples, axis=0)
        mu1 = self._mean
        cov0 = jnp.cov(samples.T)
        cov1 = self._cov
        
        w2d = 0.5 * (jnp.linalg.norm(mu0 - mu1)**2 + \
            jnp.trace(cov0 + cov1 - 2 * jax.scipy.linalg.sqrtm(cov0 @ cov1).\
                astype(jnp.float32)))
        
        return w2d
    
    def plot(self, ax, xlim, ylim, step):
        xx = np.arange(-xlim, xlim, step)
        yy = np.arange(-ylim, ylim, step)
        Xbb, Ybb = np.meshgrid(xx, yy)
        Zbb = self.pdf(Xbb, Ybb)
        ax.contourf(Xbb, Ybb, Zbb)

    
    
    
        
        

    