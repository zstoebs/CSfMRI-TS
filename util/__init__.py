"""
author: Zach Stoebner

Utilities for compressed sensing fMRI time series.
"""

import numpy as np
import cvxpy as cvx
import math

def mse(true, pred):
    """
    Compute mean squared error.

    Params:
        true = true signal
        pred = predicted signal

    Returns:
        mse scalar
    """
    return np.mean(np.square(true - pred))

def rmse(true, pred):
    """
    Compute root mean squared error.

    Params:
        true = true signal
        pred = predicted signal

    Returns:
        rmse scalar
    """
    return np.sqrt(mse(true, pred))

def psnr(true, pred):
    """
    Compute peak signal-to-noise ratio for 1D signals.

    Params:
        true = true signal
        pred = predicted signal

    Returns:
        psnr scalar
    """
    MSE = mse(true, pred)
    return 100 if MSE == 0 else 20 * math.log10(true.max() / math.sqrt(MSE))

def scale_fft(ft, N):
    """
    Scales FFT sequence to visualize for N samples. 
    https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
    
    Params:
        ft = FFT sequence
        N = sample count
    
    Returns:
        Scaled FFT sequence with N//2 entries 
    """
    return 2.0/N * np.abs(ft[:N//2])

def double_gamma_HRF(TR, tmax=30):
    """
    Construct a hemodynamic response function (HRF) based on a double-gamma fit in Glover et al. 1999, "Deconvolution
    of impulse response in event-related BOLD fMRI", Neuroimage 9(4):416-29.

    Params:
        TR = temporal resolution at which to sample in tmax
        tmax = maximum duration for HRF

    Returns:
        t = time course underlying HRF
        h = HRF sequence along t
        nyquist = Nyquist sampling rate for fMRI volumes
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

    nyquist = 2*(1/tmax)  # double the frequency of 1 HRF at least

    return t, h, nyquist


def create_task_impulse(nframes, onsets, durations):
    """
    Create a task impulse function.

    Params:
        nframes = number of frames, i.e., length of fMRI
        onsets = list of onset frame indices
        durations = list of durations in terms of frames

    Returns:
        impulse = task impulse function
    """

    assert len(onsets) == len(durations), 'Each onset should have a corresponding duration.'

    impulse = np.zeros(nframes)
    for onset, duration in zip(onsets, durations):
        start = int(onset)
        end = int(onset + duration)
        impulse[start:end] = 1

    return impulse

def CS_L1_opt(M, y, verbose=True):
    """
    Perform convex L1 optimization for sparse signal reconstruction to solve the compressed sensing problem:
    min ||x||_1 s.t. y = M @ x 
    
    Params: 
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
