import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import cv2

### 2D image reconstruction
# http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

### Part 1: load image
# read original image and downsize for speed
# https://www.sothebys.com/en/buy/auction/2019/prints-multiples-day-sale/m-c-escher-waterfall-bklw-439
Xorig = cv2.imread('Escher_Waterfall.jpg', cv2.IMREAD_GRAYSCALE) # read in grayscale
X = spimg.zoom(Xorig, 0.04)
ny,nx = X.shape

plt.figure()
plt.subplot(131)
plt.imshow(Xorig, cmap='gray')
plt.title('original image')

plt.subplot(132)
plt.imshow(X, cmap='gray')
plt.title('downsized image')

plt.subplot(133)
plt.imshow(dct2(X), cmap='gray')
plt.title('DCT downsized')

plt.savefig('in_image.png')

# Part 2: construct compressed sensing problem for a 2D image
# extract small sample of signal
k = round(nx * ny * 0.5) # 50% sample
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = X.T.flat[ri]
# b = np.expand_dims(b, axis=1)  # BUG

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:] # same as phi times kron

# do L1 optimization
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A@vx == b]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

# reconstruct signal
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

# confirm solution
if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning: values at sample indices don\'t match original.')

# create images of mask (for visualization)
mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

plt.figure()
plt.subplot(141)
plt.imshow(X, cmap='gray')
plt.title('original')

plt.subplot(142)
plt.imshow(mask, cmap='gray')
plt.title('mask')

plt.subplot(143)
plt.imshow(Xm, cmap='gray')
plt.title('downsampled')

plt.subplot(144)
plt.imshow(Xa, cmap='gray')
plt.title('reconstructed')


plt.subplots_adjust(left=0.1, right=0.9, top = 0.90, bottom=0.1, hspace=0.5, wspace=0.3)
plt.savefig('image_recon.png')
