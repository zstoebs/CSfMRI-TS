"""
author: Zach Stoebner

Utilities for compressed sensing fMRI time series.
"""

import numpy as np
import math
import scipy.fftpack as spfft

eps = np.finfo(float).eps


def nyquist_rate(ft, xf):
    """
    Computes Nyquist rate.

    Parameters:
        ft = Fourier transform sequence
        xf = corresponding frequency domain

    Returns:
        scalar Nyquist rate
    """
    return 2 * xf[np.argwhere(np.abs(ft) > eps).max()]


def mse(true, pred):
    """
    Compute mean squared error.

    Parameters:
        true = true signal
        pred = predicted signal

    Returns:
        mse scalar
    """
    return np.mean(np.square(true - pred))


def rmse(true, pred):
    """
    Compute root mean squared error.

    Parameters:
        true = true signal
        pred = predicted signal

    Returns:
        scalar RMSE
    """
    return np.sqrt(mse(true, pred))


def psnr(true, pred):
    """
    Compute peak signal-to-noise ratio for 1D signals.

    Parameters:
        true = true signal
        pred = predicted signal

    Returns:
        scalar PSNR
    """
    MSE = mse(true, pred)
    return 100 if MSE == 0 else 20 * math.log10(true.max() / math.sqrt(MSE))


def scale_fft(ft, N):
    """
    Scales FFT sequence to visualize for N samples. 
    https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
    
    Parameters:
        ft = FFT sequence
        N = sample count
    
    Returns:
        scaled FFT sequence with N//2 entries
    """
    return 2.0 / N * np.abs(ft[:N // 2])


def double_gamma_HRF(TR, tmax=30):
    """
    Construct a hemodynamic response function (HRF) based on a double-gamma fit in Glover et al. 1999, "Deconvolution
    of impulse response in event-related BOLD fMRI", Neuroimage 9(4):416-29.

    Parameters:
        TR = temporal resolution at which to sample in tmax
        tmax = maximum duration for HRF, default = 30

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
    h1 = t ** (n1) * np.exp(-t / t1)
    h2 = t ** (n2) * np.exp(-t / t2)

    # hrf as a function of two gammas
    h = h1 / np.max(h1) - a2 * h2 / np.max(h2)
    h /= np.max(h)

    hfft = scale_fft(spfft.fft(h), tmax)
    xf = np.abs(np.fft.fftfreq(tmax, TR))[:tmax // 2]
    nyquist = nyquist_rate(hfft, xf)

    return t, h, nyquist


def create_task_impulse(nframes, onsets, durations):
    """
    Create a task impulse function.

    Parameters:
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
