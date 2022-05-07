"""
author: Zach Stoebner
EECE 8396 S22

Optimization-based compressed sensing of fMRI time series.
"""

import matplotlib.pyplot as plt
import scipy.stats as spstat
import pandas as pd
import nibabel as nb
import argparse
from lbfgs import fmin_lbfgs as owlqn
from util import *

from util.opt import f_owlqn, progress, CS_L1_opt


def get_args():
    parser = argparse.ArgumentParser(description='Input args for opt-based CSfMRI-TS',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fmri', '-f', type=str, default=False,
                        help='fMRI .nii file to load')
    parser.add_argument('--task', '-t', type=str, default=False,
                        help='task .tsv file to load')
    parser.add_argument('--slice', '-s', type=int, default=10,
                        help='slice to grab voxel from based on beta search')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')

    parser.add_argument('--method', '-m', type=str, default='convex',
                        help='optimization method [convex | owlqn | bsbl]. default=convex')

    return parser.parse_args()


def optCSfMRI_TS(ffmri, ftask, method='convex', slice=10, verbose=False):
    """
    Compressed sensing a voxel time series via L1 minimization through convex optimization.

    Parameters:
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
    Fs = 1 / TR  # sampling frequency
    N = img.shape[-1]
    t = np.arange(N)
    xf = np.abs(np.fft.fftfreq(N, TR))[:N // 2]  # positive frequency domain for FFT plotting

    assert slice < img.shape[-2], 'Slice index out of bounds.'

    print('Generating HRF...')
    t_hrf, hrf, nyHRF = double_gamma_HRF(TR)

    if Fs >= nyHRF:
        print('Sampled above the Nyquist rate for HRF. Rate = %.2f HZ >= %.2f Hz = Nyquist' % (Fs, nyHRF))
    else:
        print('Sampled below the Nyquist rate for HRF. Rate = %.2f HZ < %.2f Hz = Nyquist' % (Fs, nyHRF))

    plt.figure()
    plt.plot(t_hrf, hrf)
    plt.xlabel('time (s)')
    plt.title('HRF model')
    plt.savefig('results/opt/hrf.png')

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

    respfft = scale_fft(spfft.fft(response), N)
    impfft = scale_fft(spfft.fft(impulse), N)

    nyResp = nyquist_rate(respfft, xf)  # double the max frequency of the response FFT
    if Fs >= nyResp:
        print('Sampled above the Nyquist rate for TRF. Rate = %.2f HZ >= %.2f Hz = Nyquist' % (Fs, nyResp))
    else:
        print('Sampled below the Nyquist rate for TRF. Rate = %.2f HZ < %.2f Hz = Nyquist' % (Fs, nyResp))

    plt.figure(figsize=(10, 30))

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
    plt.plot(xf, respfft, label='resp FFT')
    plt.plot(xf, impfft, label='imp FFT')
    plt.xlabel('Hz')
    plt.title('Response + impulse FFT')
    plt.legend()

    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.5, wspace=0.5)
    plt.savefig('results/opt/expected.png')

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
    # Yr = Y - Yhat
    nuisance = X[:, :3] @ Beta[:3]
    Yr = Y - nuisance

    # recon images + select voxel with high beta in regressor in slice --> discover active voxel
    Yhat_img = Yhat.T.reshape(img.shape)
    Yr_img = Yr.T.reshape(img.shape)
    Beta_map = Beta[-1, :].T.reshape(img.shape[:-1])
    b10 = Beta_map[:, :, slice]
    i, j = np.unravel_index(b10.argmax(), b10.shape)

    # active voxel time series
    y = img[i, j, slice, :]
    yhat = Yhat_img[i, j, slice, :]
    yr = Yr_img[i, j, slice, :]

    fig = plt.figure(figsize=(30, 10))
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
    plt.savefig('results/opt/active.png')

    ### TRANSFORMS
    # DCTs + iDCTs
    ydct = spfft.dct(y, norm='ortho')
    yhatdct = spfft.dct(yhat, norm='ortho')
    yrdct = spfft.dct(yr, norm='ortho')

    ydcti = spfft.idct(ydct, norm='ortho', axis=0)
    yhatdcti = spfft.idct(yhatdct, norm='ortho', axis=0)
    yrdcti = spfft.idct(yrdct, norm='ortho', axis=0)

    # DCT row
    fig = plt.figure(figsize=(30, 20))
    plt.subplot(231)
    plt.plot(ydct[1:], label='Yt')
    plt.xlabel('k')
    plt.ylabel('DCT')
    plt.subplot(232)
    plt.plot(yhatdct[1:], label='Yhatt')
    plt.xlabel('k')
    plt.subplot(233)
    plt.plot(yrdct[1:], label='Yrt')
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
    plt.savefig('results/opt/dct.png')

    # FFT + iFFT
    yfft = spfft.fft(y)
    yhatfft = spfft.fft(yhat)
    yrfft = spfft.fft(yr)

    yffti = spfft.ifft(yfft)
    yhatffti = spfft.ifft(yhatfft)
    yrffti = spfft.ifft(yrfft)

    # FFT row
    fig = plt.figure(figsize=(30, 20))
    plt.subplot(231)
    plt.plot(xf[1:], scale_fft(yfft, N)[1:], label='Yt')
    plt.xlabel('Hz')
    plt.ylabel('FFT')
    plt.subplot(232)
    plt.plot(xf[1:], scale_fft(yhatfft, N)[1:], label='Yhatt')
    plt.xlabel('Hz')
    plt.subplot(233)
    plt.plot(xf[1:], scale_fft(yrfft, N)[1:], label='Yrt')
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
    plt.savefig('results/opt/fft.png')

    ### L1 CONVEX OPT
    RMSEs = []
    PSNRs = []
    levels = np.arange(0.1, 1, 0.1)  # undersample at 10% levels + sense via convex opt
    for level in levels:
        m = int(level * N)
        ri = np.random.choice(N, m, replace=False)  # random sample of indices
        ri.sort()  # sorting not strictly necessary, but convenient for plotting

        y1 = y[ri]
        yhat1 = yhat[ri]
        yr1 = yr[ri]
        t1 = t[ri]

        # L1 optimizations + recon
        if method.lower() == 'convex':
            A = spfft.idct(np.identity(N), norm='ortho', axis=0)  # inverse discrete cosine transform
            M = A[ri]

            x = CS_L1_opt(M, y1, verbose=verbose)
            xhat = CS_L1_opt(M, yhat1, verbose=verbose)
            xr = CS_L1_opt(M, yr1, verbose=verbose)
        elif method.lower() == 'owlqn':
            x = owlqn(f_owlqn, y, progress=progress, orthantwise_c=5,
                         line_search='wolfe', args=(y1, ri))
            xhat = owlqn(f_owlqn, yhat, progress=progress, orthantwise_c=5,
                      line_search='wolfe', args=(yhat1, ri))
            xr = owlqn(f_owlqn, yr, progress=progress, orthantwise_c=5,
                      line_search='wolfe', args=(yr1, ri))
        else:
            raise ValueError('Unknown method: ', method)

        sig = spfft.idct(x, norm='ortho', axis=0)  # fully-sampled inverse cosine transform of input
        sighat = spfft.idct(xhat, norm='ortho', axis=0)
        sigr = spfft.idct(xr, norm='ortho', axis=0)

        y_rmse = rmse(y, sig)
        yhat_rmse = rmse(yhat, sighat)
        yr_rmse = rmse(yr, sigr)
        RMSEs += [(y_rmse, yhat_rmse, yr_rmse)]

        y_psnr = psnr(y, sig)
        yhat_psnr = psnr(yhat, sighat)
        yr_psnr = psnr(yr, sigr)
        PSNRs += [(y_psnr, yhat_psnr, yr_psnr)]

        # plot sensing results
        fig = plt.figure(figsize=(20, 20))

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
        plt.plot(ydct[1:], label='Yt')
        plt.plot(x[1:], label='x')
        plt.xlabel('k')
        plt.ylabel('DCT')
        plt.legend()
        plt.subplot(235)
        plt.plot(yhatdct[1:], label='Yhatt')
        plt.plot(xhat[1:], label='xhat')
        plt.xlabel('k')
        plt.legend()
        plt.subplot(236)
        plt.plot(yrdct[1:], label='Yrt')
        plt.plot(xr[1:], label='xr')
        plt.xlabel('k')
        plt.legend()

        fig.suptitle('Undersampling signals at %.1f' % level)
        plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
        plt.savefig('results/opt/%s/sensing_at_%.1f.png' % (method, level))

    # error curve with Nyquist threshold
    RMSEs = np.asarray(RMSEs)
    PSNRs = np.asarray(PSNRs)
    nyHRFPercent = TR * nyHRF  # length*rate / N = N*TR*nyquist / N = TR*nyquist
    nyRespPercent = TR * nyResp

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(levels, RMSEs[:, 0], label='Y')
    plt.plot(levels, RMSEs[:, 1], label='Yhat')
    plt.plot(levels, RMSEs[:, 2], label='Yr')
    plt.axvline(x=nyHRFPercent, label='%% HRF Nyquist = %.2f' % nyHRFPercent, c='c', ls='--')
    plt.axvline(x=nyRespPercent, label='%% Resp Nyquist = %.2f' % nyRespPercent, c='m', ls='--')
    plt.xlabel('Percent sampled')
    plt.ylabel('PSNR')
    plt.legend()
    plt.subplot(122)
    plt.plot(levels, PSNRs[:, 0], label='Y')
    plt.plot(levels, PSNRs[:, 1], label='Yhat')
    plt.plot(levels, PSNRs[:, 2], label='Yr')
    plt.axvline(x=nyHRFPercent, label='%% HRF Nyquist = %.2f' % nyHRFPercent, c='c', ls='--')
    plt.axvline(x=nyRespPercent, label='%% Resp Nyquist = %.2f' % nyRespPercent, c='m', ls='--')
    plt.xlabel('Percent sampled')
    plt.ylabel('PSNR')
    plt.legend()

    fig.suptitle('RMSE + PSNR of time series recovery')
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.4, wspace=0.5)
    plt.savefig('results/opt/%s/rmse+psnr.png' % method)


def main(**kwargs):
    args = get_args()

    ###
    print('Compressed sensing time series...')
    optCSfMRI_TS(ffmri=args.fmri, ftask=args.task, method=args.method, slice=args.slice, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    main()
