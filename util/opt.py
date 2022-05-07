"""
author: Zach Stoebner

Utilities for opt-based compressed sensing fMRI time series.
"""

import numpy as np
import cvxpy as cvx
import scipy.fftpack as spfft


def CS_L1_opt(M, y, verbose=True):
    """
    Perform convex L1 optimization for sparse signal reconstruction to solve the compressed sensing problem:
    min ||x||_1 s.t. y = M @ x

    Parameters:
        M = undersampled sparese measurement matrix reflecting the basis of observations
        y = undersampled observations s.t. m < n for measurement matrix

    Returns:
        x_rec = recovered signal
    """

    # perform optimization
    x = cvx.Variable(M.shape[-1])
    objective = cvx.Minimize(cvx.norm(x, 1))
    constraints = [M @ x == y]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=verbose)

    # reconstruct signals
    x_rec = np.array(x.value)
    x_rec = np.squeeze(x_rec)

    return x_rec


def progress(x, g, fx, xnorm, gnorm, step, k, ls, *args):
    """Display the current iteration.
    """
    if k % 10 == 0:
        print('{}         Xnorm = {:.3f}            Gnorm = {:.3f}'.format(k, xnorm, gnorm))
    return 0  # expects return of 0 for exit success


def f_owlqn(x, g, *args):
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
    b = args[0]
    ri = args[1]

    # compute the transform of the current guess
    Ax_full = spfft.idct(x, norm='ortho', axis=0)

    # get the unknown samples
    Ax = Ax_full[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared loss
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank
    Axb2 = np.zeros(x.shape)
    Axb2[ri] = Axb

    # A'(Ax-b) is just the dct of Axb
    AtAxb2 = 2 * spfft.dct(Axb2, norm='ortho')
    AtAxb = AtAxb2.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx
