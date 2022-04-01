"""
author: Zach Stoebner
EECE 8396 S22

Optimization-based compressed sensing of fMRI time series.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.stats as spstat
import pandas as pd
import cv2
from lbfgs import fmin_lbfgs as owlqn  # pip install pylbfgs or (deprecated) https://bitbucket.org/rtaylor/pylbfgs/src/master/
import time
from datetime import timedelta
import nibabel as nb
from util import double_gamma_HRF, create_task_impulse, CS_L1_opt

def CS_Ex4(ffmri, ftask, slice=10, verbose=False):
    """
    Compressed sensing on data from class exercise.

    Params:
        ffmri = fMRI filename
        ftask = task spreadsheet filename
        slice = slice number for voxel analysis
        verbose = true to display optimization summaries
    
    Returns:
        void
    """

    fmri = nb.load(ffmri)
    img = fmri.get_fdata()
    hdr = fmri.header
    TR = hdr['pixdim'][4]
    nframes = img.shape[-1]
    t = np.arange(nframes)

    print('Generating HRF...')
    t_hrf, hrf, nyHRF = double_gamma_HRF(TR)

    rate = 1/TR
    if rate >= nyHRF:
        print('Sampled above the Nyquist rate for HRF. Rate = %.2f HZ >= %.2f Hz = Nyquist' % (rate, nyHRF))
    else:
        print('Sampled below the Nyquist rate for HRF. Rate = %.2f HZ < %.2f Hz = Nyquist' % (rate, nyHRF))

    plt.figure()
    plt.plot(t_hrf,hrf)
    plt.xlabel('time (s)')
    plt.title('HRF model')
    plt.savefig('results/hrf.png')

    # response function on 20-second task, every 60 secs, starting at 30 secs
    task = pd.read_csv(ftask, sep='\t')
    onsets = task['onset'].to_numpy()
    durations = task['duration'].to_numpy()
    impulse = create_task_impulse(nframes, onsets // TR, durations // TR)
    response = np.convolve(impulse, hrf, mode='full')  # mode = 'full', 'valid', 'same'
    response = response[:nframes]

    respt = spfft.dct(response, norm='ortho')
    impt = spfft.dct(impulse, norm='ortho')

    plt.figure()

    plt.subplot(211)
    plt.plot(response, label='response')
    plt.plot(impulse, label='impulse')
    plt.xlabel('frame')
    plt.title('Expected response given impulse')
    plt.legend()
    plt.subplot(212)
    plt.plot(respt, label='resp DCT')
    plt.plot(impt, label='imp DCT')
    plt.xlabel('k')
    plt.legend()
    plt.title('Response + impulse DCT')

    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.5, wspace=0.5)
    plt.savefig('results/expected.png')

    # design matrix
    lin = t.copy()
    quad = lin ** 2
    X = np.vstack([lin, quad, response]).T
    X = np.hstack([np.ones((nframes, 1)), spstat.zscore(X, axis=0)])
    Y = np.transpose(img, (3, 0, 1, 2)).reshape(nframes, -1)

    # compute coefs + residual
    Beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    Yhat = X @ Beta
    Yr = Y - Yhat

    # recon images + select voxel with high beta in regressor in slice
    Yhat_img = Yhat.T.reshape(img.shape)
    Yr_img = Yr.T.reshape(img.shape)
    Beta_map = Beta[-1, :].T.reshape(img.shape[:-1])
    b10 = Beta_map[:, :, slice]
    i,j = np.unravel_index(b10.argmax(), b10.shape)

    # active voxel time series
    y = img[i, j, slice, :]
    yhat = Yhat_img[i, j, slice, :]
    yr = Yr_img[i, j, slice, :]

    # DCTs of time series
    yt = spfft.dct(y, norm='ortho')
    yhatt = spfft.dct(yhat, norm='ortho')
    yrt = spfft.dct(yr, norm='ortho')

    # iDCTs
    yti = spfft.idct(yt, norm='ortho', axis=0)
    yhatti = spfft.idct(yhatt, norm='ortho', axis=0)
    yrti = spfft.idct(yrt, norm='ortho', axis=0)

    fig = plt.figure(figsize=(20,20))

    # time series row
    plt.subplot(331)
    plt.plot(y, label='Y')
    plt.xlabel('TR')
    plt.ylabel('Signal')
    plt.gca().set_title('Y')
    plt.subplot(332)
    plt.plot(yhat, label='Yhat')
    plt.xlabel('TR')
    plt.gca().set_title('Yhat')
    plt.subplot(333)
    plt.plot(yr, label='Yr')
    plt.xlabel('TR')
    plt.gca().set_title('Yr')

    # DCT row
    plt.subplot(334)
    plt.plot(yt, label='Yt')
    plt.xlabel('k')
    plt.ylabel('DCT')
    plt.subplot(335)
    plt.plot(yhatt, label='Yhatt')
    plt.xlabel('k')
    plt.subplot(336)
    plt.plot(yrt, label='Yrt')
    plt.xlabel('k')

    # iDCT row
    plt.subplot(337)
    plt.plot(yti, label='Yti')
    plt.xlabel('TR')
    plt.ylabel('iDCT')
    plt.subplot(338)
    plt.plot(yhatti, label='Yhatti')
    plt.xlabel('TR')
    plt.subplot(339)
    plt.plot(yrti, label='Yrti')
    plt.xlabel('TR')

    fig.suptitle('Active voxel time series + FFTs + iFFTs')
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
    plt.savefig('results/active_ts+fft+ifft.png')

    ### L1 convex optimization compressed sensing fMRI time series at varying granularity

    # undersample at 10% levels + sense via convex opt
    errors = []
    levels = np.arange(0.1,1,0.1)
    A = spfft.idct(np.identity(nframes), norm='ortho', axis=0)  # inverse discrete cosine transform
    for level in levels:
        m = int(level * nframes)
        ri = np.random.choice(nframes, m, replace=False)  # random sample of indices
        ri.sort()  # sorting not strictly necessary, but convenient for plotting

        y1 = y[ri]
        yhat1 = yhat[ri]
        yr1 = yr[ri]
        t1 = t[ri]
        M = A[ri]

        # L1 optimizations + recon
        x = CS_L1_opt(M, y1, verbose=verbose)
        xhat = CS_L1_opt(M, yhat1, verbose=verbose)
        xr = CS_L1_opt(M, yr1, verbose=verbose)

        sig = spfft.idct(x, norm='ortho', axis=0)  # fully-sampled inverse cosine transform of input
        sighat = spfft.idct(xhat, norm='ortho', axis=0)
        sigr = spfft.idct(xr, norm='ortho', axis=0)
        
        y_err = np.mean(np.square(y - sig))
        yhat_err = np.mean(np.square(yhat - sighat))
        yr_err = np.mean(np.square(yr - sigr))
        errors += [(y_err, yhat_err, yr_err)]

        # plot sensing results
        fig = plt.figure(figsize=(20,20))

        # signal: original + samples + recon
        plt.subplot(231)
        plt.plot(t, y, label='Y')
        plt.plot(t1, y1, 'ro', label='samples')
        plt.plot(t, sig, label='recon')
        plt.xlabel('TR')
        plt.title('Y')
        plt.legend()
        plt.subplot(232)
        plt.plot(t, yhat, label='Yhat')
        plt.plot(t1, yhat1, 'ro', label='samples')
        plt.plot(t, sighat, label='recon')
        plt.xlabel('TR')
        plt.title('Yhat')
        plt.legend()
        plt.subplot(233)
        plt.plot(t, yr, label='Yr')
        plt.plot(t1, yr1, 'ro', label='samples')
        plt.plot(t, sigr, label='recon')
        plt.xlabel('TR')
        plt.title('Yr')
        plt.legend()

        # spectral: original + recon
        plt.subplot(234)
        plt.plot(yt, label='Yt')
        plt.plot(x, label='x')
        plt.xlabel('k')
        plt.ylabel('DCT')
        plt.legend()
        plt.subplot(235)
        plt.plot(yhatt, label='Yhatt')
        plt.plot(xhat, label='xhat')
        plt.xlabel('k')
        plt.legend()
        plt.subplot(236)
        plt.plot(yrt, label='Yrt')
        plt.plot(xr, label='xr')
        plt.xlabel('k')
        plt.legend()

        fig.suptitle('Undersampling signals at %.1f' % level)
        plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
        plt.savefig('results/sensing_at_%.1f.png' % level)

    # error curve
    errors = np.asarray(errors)
    nyHRFPercent = TR*nyHRF  # length*rate / nframes = nframes*TR*nyquist / nframes = TR*nyquist
    nyRespPercent = TR*nyResp

    plt.figure()
    plt.plot(levels, errors[:,0], label='Y')
    plt.plot(levels, errors[:, 1], label='Yhat')
    plt.plot(levels, errors[:, 2], label='Yr')
    plt.axvline(x=nyHRFPercent, label='% HRF Nyquist', c='c', ls='--')
    plt.axvline(x=nyRespPercent, label='% Resp Nyquist', c='m', ls='--')
    plt.xlabel('Percent sampled')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('results/error.png')

def main(**kwargs):
    print('Executing...')

    ###
    print('Compressed sensing Ex4...')
    CS_Ex4('data/fmri_blockDes.nii.gz', 'data/task-checkerboard_events.tsv', slice=10)

    return 0

if __name__ == "__main__":
    main()
