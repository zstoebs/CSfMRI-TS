"""
author: Zach Stoebner
EECE 8396 S22

Optimization-based compressed sensing of fMRI time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.stats as spstat
import pandas as pd
import cv2
import nibabel as nb
from util import double_gamma_HRF, create_task_impulse, CS_L1_opt, rmse, scale_fft, psnr

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

    ### SETUP
    fmri = nb.load(ffmri)
    img = fmri.get_fdata()
    hdr = fmri.header
    TR = hdr['pixdim'][4]
    N = img.shape[-1]
    t = np.arange(N)
    xf = np.linspace(0.0, 1.0 / (2.0 * TR), N // 2)  # frequency domain for FFT plotting

    print('Generating HRF...')
    t_hrf, hrf, nyHRF = double_gamma_HRF(TR)

    Fs = 1/TR
    if Fs >= nyHRF:
        print('Sampled above the Nyquist rate for HRF. Rate = %.2f HZ >= %.2f Hz = Nyquist' % (Fs, nyHRF))
    else:
        print('Sampled below the Nyquist rate for HRF. Rate = %.2f HZ < %.2f Hz = Nyquist' % (Fs, nyHRF))

    plt.figure()
    plt.plot(t_hrf,hrf)
    plt.xlabel('time (s)')
    plt.title('HRF model')
    plt.savefig('results/hrf.png')

    # response function on 20-second task, every 60 secs, starting at 30 secs
    task = pd.read_csv(ftask, sep='\t')
    onsets = task['onset'].to_numpy()
    durations = task['duration'].to_numpy()
    impulse = create_task_impulse(N, onsets // TR, durations // TR)
    response = np.convolve(impulse, hrf, mode='full')  # mode = 'full', 'valid', 'same'
    response = response[:N]

    # transforms + Nyquist rate
    respdct = spfft.dct(response, norm='ortho')
    impdct = spfft.dct(impulse, norm='ortho')

    respfft = spfft.fft(response)
    impfft = spfft.fft(impulse)

    nyResp = 2 * xf[np.argmax(respfft)]  # double the max frequency of the response FFT
    if Fs >= nyResp:
        print('Sampled above the Nyquist rate for response. Rate = %.2f HZ >= %.2f Hz = Nyquist' % (Fs, nyResp))
    else:
        print('Sampled below the Nyquist rate for response. Rate = %.2f HZ < %.2f Hz = Nyquist' % (Fs, nyResp))

    plt.figure(figsize=(10,30))

    plt.subplot(311)
    plt.plot(response, label='response')
    plt.plot(impulse, label='impulse')
    plt.xlabel('frame')
    plt.title('Expected response given impulse')
    plt.legend()
    plt.subplot(312)
    plt.plot(respdct, label='resp DCT')
    plt.plot(impdct, label='imp DCT')
    plt.xlabel('k')
    plt.title('Response + impulse DCT')
    plt.legend()
    plt.subplot(313)
    plt.plot(xf, scale_fft(respfft, N), label='resp FFT')
    plt.plot(xf, scale_fft(impfft, N), label='imp FFT')
    plt.xlabel('Hz')
    plt.title('Response + impulse FFT')
    plt.legend()

    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.5, wspace=0.5)
    plt.savefig('results/expected.png')

    ### GLM
    # design matrix
    lin = t.copy()
    quad = lin ** 2
    X = np.vstack([lin, quad, response]).T
    X = np.hstack([np.ones((N, 1)), spstat.zscore(X, axis=0)])
    Y = np.transpose(img, (3, 0, 1, 2)).reshape(N, -1)

    # compute coefs + residual
    Beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    Yhat = X @ Beta
    Yr = Y - Yhat

    # recon images + select voxel with high beta in regressor in slice --> discover active voxel
    Yhat_img = Yhat.T.reshape(img.shape)
    Yr_img = Yr.T.reshape(img.shape)
    Beta_map = Beta[-1, :].T.reshape(img.shape[:-1])
    b10 = Beta_map[:, :, slice]
    i,j = np.unravel_index(b10.argmax(), b10.shape)

    # active voxel time series
    y = img[i, j, slice, :]
    yhat = Yhat_img[i, j, slice, :]
    yr = Yr_img[i, j, slice, :]

    fig = plt.figure(figsize=(30,10))
    plt.subplot(131)
    plt.plot(y, label='Y')
    plt.xlabel('TR')
    plt.ylabel('Signal')
    plt.title('Y')
    plt.subplot(132)
    plt.plot(yhat, label='Yhat')
    plt.xlabel('TR')
    plt.title('Yhat')
    plt.subplot(133)
    plt.plot(yr, label='Yr')
    plt.xlabel('TR')
    plt.title('Yr')

    fig.suptitle('Active voxel time series')
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
    plt.savefig('results/active.png')

    ### TRANSFORMS
    # DCTs + iDCTs
    ydct = spfft.dct(y, norm='ortho')
    yhatdct = spfft.dct(yhat, norm='ortho')
    yrdct = spfft.dct(yr, norm='ortho')

    ydcti = spfft.idct(ydct, norm='ortho', axis=0)
    yhatdcti = spfft.idct(yhatdct, norm='ortho', axis=0)
    yrdcti = spfft.idct(yrdct, norm='ortho', axis=0)

    # DCT row
    fig = plt.figure(figsize=(30,20))
    plt.subplot(231)
    plt.plot(ydct, label='Yt')
    plt.xlabel('k')
    plt.ylabel('DCT')
    plt.subplot(232)
    plt.plot(yhatdct, label='Yhatt')
    plt.xlabel('k')
    plt.subplot(233)
    plt.plot(yrdct, label='Yrt')
    plt.xlabel('k')

    # iDCT row
    plt.subplot(234)
    plt.plot(ydcti, label='Yti')
    plt.xlabel('TR')
    plt.ylabel('iDCT')
    plt.subplot(235)
    plt.plot(yhatdcti, label='Yhatti')
    plt.xlabel('TR')
    plt.subplot(236)
    plt.plot(yrdcti, label='Yrti')
    plt.xlabel('TR')

    fig.suptitle('Active DCTs + iDCTs')
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
    plt.savefig('results/dct.png')

    # FFT + iFFT
    yfft = spfft.fft(y)
    yhatfft = spfft.fft(yhat)
    yrfft = spfft.fft(yr)

    yffti = spfft.ifft(yfft)
    yhatffti = spfft.ifft(yhatfft)
    yrffti = spfft.ifft(yrfft)

    # FFT row
    fig = plt.figure(figsize=(30,20))
    plt.subplot(231)
    plt.plot(xf, scale_fft(yfft, N), label='Yt')
    plt.xlabel('Hz')
    plt.ylabel('FFT')
    plt.subplot(232)
    plt.plot(xf, scale_fft(yhatfft, N), label='Yhatt')
    plt.xlabel('Hz')
    plt.subplot(233)
    plt.plot(xf, scale_fft(yrfft, N), label='Yrt')
    plt.xlabel('Hz')

    # iFFT row
    plt.subplot(234)
    plt.plot(yffti.real, label='Yti')
    plt.xlabel('TR')
    plt.ylabel('iFFT')
    plt.subplot(235)
    plt.plot(yhatffti.real, label='Yhatti')
    plt.xlabel('TR')
    plt.subplot(236)
    plt.plot(yrffti.real, label='Yrti')
    plt.xlabel('TR')

    fig.suptitle('Active FFTs + iFFTs')
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
    plt.savefig('results/fft.png')

    ### L1 CONVEX OPT
    errors = []
    levels = np.arange(0.1,1,0.1) # undersample at 10% levels + sense via convex opt
    A = spfft.idct(np.identity(N), norm='ortho', axis=0)  # inverse discrete cosine transform
    for level in levels:
        m = int(level * N)
        ri = np.random.choice(N, m, replace=False)  # random sample of indices
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
        
        y_err = psnr(y, sig)
        yhat_err = psnr(yhat, sighat)
        yr_err = psnr(yr, sigr)
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
        plt.plot(ydct, label='Yt')
        plt.plot(x, label='x')
        plt.xlabel('k')
        plt.ylabel('DCT')
        plt.legend()
        plt.subplot(235)
        plt.plot(yhatdct, label='Yhatt')
        plt.plot(xhat, label='xhat')
        plt.xlabel('k')
        plt.legend()
        plt.subplot(236)
        plt.plot(yrdct, label='Yrt')
        plt.plot(xr, label='xr')
        plt.xlabel('k')
        plt.legend()

        fig.suptitle('Undersampling signals at %.1f' % level)
        plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
        plt.savefig('results/sensing_at_%.1f.png' % level)

    # error curve with Nyquist threshold
    errors = np.asarray(errors)
    nyHRFPercent = TR*nyHRF  # length*rate / N = N*TR*nyquist / N = TR*nyquist
    nyRespPercent = TR*nyResp

    plt.figure()
    plt.plot(levels, errors[:,0], label='Y')
    plt.plot(levels, errors[:, 1], label='Yhat')
    plt.plot(levels, errors[:, 2], label='Yr')
    # plt.axvline(x=nyHRFPercent, label='% HRF Nyquist', c='c', ls='--')
    # plt.axvline(x=nyRespPercent, label='% Resp Nyquist', c='m', ls='--')
    plt.xlabel('Percent sampled')
    plt.ylabel('PSNR')
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
