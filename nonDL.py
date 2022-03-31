"""
author: Zach Stoebner
EECE 8396 S22

Non-deep learning compressed sensing of fMRI time series.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.stats as spstat
import cvxpy as cvx
import cv2
from lbfgs import fmin_lbfgs as owlqn  # pip install pylbfgs or (deprecated) https://bitbucket.org/rtaylor/pylbfgs/src/master/
import time
from datetime import timedelta
import nibabel as nb
from util import double_gamma_HRF, create_task_impulse

def CS_Ex4(fmri, hrf, slice=10):
    """
    Compressed sensing on data from class exercise.

    Params:
    img = nifti fMRI
    hrf = hemodynamic response function
    slice = slice number for voxel analysis
    """

    img = fmri.get_fdata()
    hdr = fmri.header
    TR = hdr['pixdim'][4]
    nframes = img.shape[-1]

    # response function on 20-second task, every 60 secs, starting at 30 secs
    onsets = np.arange(30, 270, 60)
    durations = np.ones(onsets.shape) * 20
    impulse = create_task_impulse(nframes, onsets // TR, durations // TR)
    response = np.convolve(impulse, hrf, mode='full')  # mode = 'full', 'valid', 'same'
    response = response[:nframes]

    plt.figure()
    plt.plot(response, label='response')
    plt.plot(impulse, label='impulse')
    plt.xlabel('frame')
    plt.title('expected response function')
    plt.legend()
    plt.savefig('results/expected.png')

    # design matrix
    lin = np.arange(nframes)
    quad = lin ** 2
    X = np.vstack([lin, quad, response]).T
    X = np.hstack([np.ones((nframes, 1)), spstat.zscore(X, axis=1)])
    Y = np.transpose(img, (3, 0, 1, 2)).reshape(nframes, -1)

    # compute coefs
    Beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    Yhat = X @ Beta
    Yr = Y - Yhat

    # recon images + select voxel with high beta in regressor in slice
    Yhat_img = Yhat.T.reshape(img.shape)
    Yr_img = Yr.T.reshape(img.shape)
    Beta_map = Beta[-1, :].T.reshape(img.shape[:-1])
    b10 = Beta_map[:, :, slice]
    x,y = np.unravel_index(b10.argmax(), b10.shape)

    plt.figure()
    plt.plot(Yhat_img[x,y,slice,:], label='Yhat')
    plt.plot(Yr_img[x,y,slice,:], label='Yr')
    plt.plot(img[x,y,slice,:], label='Y')
    plt.legend()
    plt.savefig('results/full_ts.png')



def main(**kwargs):
    print('Executing...')

    fmri = nb.load('data/fmri_blockDes.nii.gz')
    img = fmri.get_fdata()
    hdr = fmri.header
    TR = hdr['pixdim'][4]

    print('Generating HRF...')
    t, hrf = double_gamma_HRF(TR)

    plt.figure()
    plt.plot(t,hrf)
    plt.xlabel('time (s)')
    plt.title('HRF model')
    plt.savefig('results/hrf.png')
    
    ###
    print('Compressed sensing Ex4...')
    CS_Ex4(fmri, hrf)

    return 0

if __name__ == "__main__":
    main()
