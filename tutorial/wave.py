import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx

### Signal + FFT reconstruction
# http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

x = np.sort(np.random.uniform(0, 10, 15))
y = 0.2 * x + 3.0 + 0.1 * np.random.randn(len(x))  # y = (1/5)x + 3 + \sigma

### Part 1: l1 vs l2 norms
# find L1 line fit
l1_fit = lambda x0, x, y: np.sum(np.abs(x0[0] * x + x0[1] - y))  # error function: predicted - true
xopt1 = spopt.fmin(func=l1_fit, x0=[1, 1], args=(x, y))

# find L2 line fit
l2_fit = lambda x0, x, y: np.sum(np.power(x0[0] * x + x0[1] - y, 2))  # error function: predicted - true
xopt2 = spopt.fmin(func=l2_fit, x0=[1, 1], args=(x, y))

# plot initial points as blue dots, l1 as dashed, and l2 as solid
plt.figure()
plt.plot(x,y,'bo',label='data')
plt.plot(x,xopt1[0]*x+xopt1[1],'--', label='L1')
plt.plot(x,xopt2[0]*x+xopt2[1],'-', label='L2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('No outliers')
plt.legend()
# plt.show()
plt.savefig('no_outliers.png')

### Part 2: adding outliers to l1 vs l2 line fits
# adjust data by adding outlyers
y2 = y.copy()
y2[3] += 4
y2[13] -= 3

# refit the lines
xopt12 = spopt.fmin(func=l1_fit, x0=[1, 1], args=(x, y2))
xopt22 = spopt.fmin(func=l2_fit, x0=[1, 1], args=(x, y2))

plt.figure()
plt.plot(x,y2,'bo',label='data')
plt.plot(x,xopt12[0]*x+xopt12[1],'--', label='L1')
plt.plot(x,xopt22[0]*x+xopt22[1],'-', label='L2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('With outliers')
plt.legend()
# plt.show()
plt.savefig('with_outliers.png')

### Part 3: simple wave reconstruction
n = 5000
t = np.linspace(0, 1/8, n)
waves = lambda t: np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
y = waves(t)
t1 = t[:500]  # np.linspace(0,0.02,n) # smaller scale to see
y1 = waves(t1)
yt = spfft.dct(y, norm='ortho')

plt.figure()
plt.subplot(221)
plt.plot(t,y)
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(222)
plt.plot(t1,y1)
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(223)
plt.plot(np.arange(n),yt)
plt.xlabel('k')
plt.ylabel('yt')

plt.subplot(224)
plt.plot(np.arange(500),yt[:500])
plt.xlabel('k')
plt.ylabel('yt')

plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=0.5, wspace=0.4)
plt.savefig('waves.png')

### Part 4: undersample the waves
# extract small sample of signal
m = 500 # 10% sample
ri = np.random.choice(n, m, replace=False) # random sample of indices
ri.sort() # sorting not strictly necessary, but convenient for plotting
t2 = t[ri]
y2 = y[ri]

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(t,y)
plt.plot(t2,y2,'ro')
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(122)
plt.plot(t2,y2,'r')
plt.xlabel('t')
plt.ylabel('y')

plt.subplots_adjust(left=0.1, right=0.9, top = 0.90, bottom=0.1, hspace=0.5, wspace=0.2)
plt.savefig('undersampled.png')

### Part 5: convex optimization minimizing L1 norm to observations to solve inverse problem
# create idct matrix operator
A = spfft.idct(np.identity(n), norm='ortho', axis=0) # inverse discrete cosine transform
A = A[ri] # **undersampled

# do L1 optimization
vx = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A@vx == y2]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

# reconstruct signal
x_rec = np.array(vx.value)
x_rec = np.squeeze(x_rec)
sig = spfft.idct(x_rec, norm='ortho', axis=0) # fully-sampled inverse cosine transform of input

plt.figure()
plt.subplot(221)
plt.plot(np.arange(m),yt[:m])
plt.xlabel('k')
plt.ylabel('yt')

plt.subplot(222)
plt.plot(np.arange(m),x_rec[:m])
plt.xlabel('k')
plt.ylabel('yt')

plt.subplot(223)
plt.plot(t1,y1)
plt.xlabel('t')
plt.ylabel('y')

plt.subplot(224)
plt.plot(t1[:m],sig[:m])
plt.xlabel('t')
plt.ylabel('y')

plt.savefig('wave_recon.png')
