import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import cv2
from lbfgs import fmin_lbfgs as owlqn  # pip install pylbfgs or (deprecated) https://bitbucket.org/rtaylor/pylbfgs/src/master/
import time
# from datetime import timedelta


### **optimized** 2D image reconstruction
# http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Display the current iteration.
    """
    if k % 10 == 0:
        print('Iteration {}         Xnorm {}            Gnorm {}'.format(k, xnorm, gnorm))
    return 0  # expects return of 0 for exit success

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, *args):

    """An in-memory evaluation callback.

    This method evaluates the objective function sum((Ax-b).^2) and its
    gradient without ever actually generating A (which can be massive).
    Our ability to do this stems from our knowledge that Ax is just the
    sampled idct2 of the spectral image (x in matrix form).
    
    Params:
    x = current guess
    g = gradients
    """

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)  # refers to b and ri defined in main loop below

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

# fractions of the scaled image to randomly sample at
sample_sizes = (0.1, 0.01)

# read original image
Xorig = cv2.imread('Escher_Waterfall.jpg')
ny,nx,nchan = Xorig.shape

# for each sample size
Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
tstart = time.time()
for i,s in enumerate(sample_sizes):

    print('Reconstructing for sample size ', s)

    # create random sampling index vector
    k = round(nx * ny * s)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

    # for each color channel
    for j in range(nchan):

        # extract channel
        X = Xorig[:,:,j].squeeze()

        # create images of mask (for visualization)
        Xm = 255 * np.ones(X.shape)
        Xm.T.flat[ri] = X.T.flat[ri]
        masks[i][:,:,j] = Xm

        # take random samples of image, store them in a vector b
        b = X.T.flat[ri].astype(float)  # referred to in evaluate()

        # perform the L1 minimization in memory
        print('L1 OWL-QN for channel ', j+1)

        """
        Params:
        f : callable(x, g, *args)
            Computes function to minimize and its gradient.
            Called with the current position x (a numpy.ndarray), a gradient
            vector g (a numpy.ndarray) to be filled in and *args.
            Must return the value at x and set the gradient vector g.
            
        x0 : array-like
            Initial values. A copy of this array is made prior to optimization.
            
            ** Init guess of Xm (masked) undersample, or even an array of ones, 
            converge to similar images.**
            
        progress : callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args),
                   optional
            If not None, called at each iteration after the call to f with the
            current values of x, g and f(x), the L2 norms of x and g, the line
            search step, the iteration number, the number of evaluations at
            this iteration and args (see below).
            If the return value from this callable is not 0 and not None,
            optimization is stopped and LBFGSError is raised.
            
        orthantwise_c: float, optional (default=0)
            Coefficient for the L1 norm of variables.
            This parameter should be set to zero for standard minimization
            problems. Setting this parameter to a positive value activates
            Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
            minimizes the objective function F(x) combined with the L1 norm |x|
            of the variables, {F(x) + C |x|}. This parameter is the coefficient
            for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
            zero, the library modifies function and gradient evaluations from
            a client program suitably; a client program thus have only to return
            the function value F(x) and gradients G(x) as usual. The default value
            is zero. 
            
            If orthantwise_c is set, then line_search cannot be the default
            and must be one of 'armijo', 'wolfe', or 'strongwolfe'.
            
        line_search: str, optional (default="default")
            The line search algorithm.
            This parameter specifies a line search algorithm to be used by the
            L-BFGS routine. Possible values are:
            - 'default': same as 'morethuente'
            - 'morethuente': Method proposed by More and Thuente
            - 'armijo': backtracking with Armijo's conditions
            - 'wolfe': backtracking with Wolfe's conditions
            - 'strongwolfe': backtracking with strong Wolfe's conditions
        """
        Xat2 = owlqn(evaluate, np.ones(Xm.shape), progress=progress, orthantwise_c=5, line_search='wolfe')  # TODO debug

        # transform the output back into the spatial domain
        Xat = Xat2.reshape(nx, ny).T # stack columns
        Xa = idct2(Xat)
        Z[i][:,:,j] = Xa.astype('uint8')

tend = time.time()
elapsed = tstart - tend
# print('Runtime: ', timedelta(seconds=elapsed))
print('Runtime: ', elapsed)

plt.figure()
plt.subplot(131)
plt.imshow(Xorig)
plt.title('original')

plt.subplot(132)
plt.imshow(masks[0])
plt.title('sample=0.1')

plt.subplot(133)
plt.imshow(Z[0])
plt.title('reconstruction')

plt.savefig('optimized_recon=0.1.png')

plt.figure()
plt.subplot(131)
plt.imshow(Xorig)
plt.title('original')

plt.subplot(132)
plt.imshow(masks[1])
plt.title('sample=0.01')

plt.subplot(133)
plt.imshow(Z[1])
plt.title('reconstruction')

plt.savefig('optimized_recon=0.01.png')
