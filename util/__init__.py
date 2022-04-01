"""
author: Zach Stoebner

Utilities for compressed sensing fMRI time series.
"""

import numpy as np
import cvxpy as cvx

def double_gamma_HRF(TR, tmax=30):
    """
    Construct a hemodynamic response function (HRF) based on a double-gamma fit in Glover et al. 1999, "Deconvolution
    of impulse response in event-related BOLD fMRI", Neuroimage 9(4):416-29.

    Params:
    TR = temporal resolution at which to sample in tmax
    tmax = maximum duration for HRF
    """

    # gamma params
    t = np.arange(0, tmax, TR)
    n1 = 5.0
    t1 = 1.1
    n2 = 12.0
    t2 = 0.9
    a2 = 0.4

    # gamma functions
    h1 = t**(n1)*np.exp(-t/t1)
    h2 = t**(n2)*np.exp(-t/t2)

    # hrf as a function of two gammas
    h = h1/np.max(h1) - a2*h2/np.max(h2)
    h /= np.max(h)

    return t, h


def create_task_impulse(nframes, onsets, durations):
    """
    Create a task impulse function.

    Params:
    nframes = number of frames, i.e., length of fMRI
    onsets = list of onset frame indices
    durations = list of durations in terms of frames
    """

    assert len(onsets) == len(durations), 'Each onset should have a corresponding duration.'

    impulse = np.zeros(nframes)
    for onset, duration in zip(onsets, durations):
        start = int(onset)
        end = int(onset + duration)
        impulse[start:end] = 1

    return impulse

def CS_L1_opt(M, y):
    """
    Perform convex L1 optimization for sparse signal reconstruction to solve the compressed sensing problem:
    min ||x||_1 s.t. y = M @ x 
    
    Params: 
    M = undersampled sparese measurement matrix reflecting the basis of observations
    y = undersampled observations s.t. m < n for measurement matrix
    """
    
    # perform optimization
    x = cvx.Variable(M.shape[-1])
    objective = cvx.Minimize(cvx.norm(x, 1))
    constraints = [M @ x == y]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    # reconstruct signals
    x_rec = np.array(x.value)
    x_rec = np.squeeze(x_rec)
    
    return x_rec