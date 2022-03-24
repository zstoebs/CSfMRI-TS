import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import cv2
from lbfgs import fmin_lbfgs as owlqn  # pip install pylbfgs or (deprecated) https://bitbucket.org/rtaylor/pylbfgs/src/master/
import time