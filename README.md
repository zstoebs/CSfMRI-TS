# CSfMRI-TS
Author: Zach Stoebner

EECE 8396 S22


## Description
Exploring compressed sensing fMRI time series.

## Parts
- [tutorial](/tutorial/): background tutorial for compressed sensing with examples for:
	1. the traditional convex optimization approach to recover a wave signal + a downsized 2D image from the discrete cosine transform, and 
	2. an L-BFGS gradient descent approach to recover a full-sized 2D image from the discrete cosine transform. 
- [opt](/optCSfMRI-TS.py): L1 minimization of voxel time series through general convex optimization formulation. Analysis of a task-based fMRI to identify the most active voxel correlated to the task and exploring sparse recovery related to the voxel activation, in the context of the Nyquist rates defined by the HRF and task response function. Using:
	1. ECOS (baseline)
	2. OWL-QN
	3. BSBL-BO

## Usage
### tutorial
- Run wave.py to reconstruct a simple waveform using L1 minimization via convex optimization. 
- Run image.py to reconstruct a downsized image using L1 minimization via convex optimization. 
- Run optimized.py to reconstruct the full-size image using OWL-QN version of the L-BFGS algorithm. 

### opt
`python optCSfMRI-TS.py -f <FMRI_FILE> -t <TASK_FILE> [-s <SLICE_INDEX>] [-m <METHOD>] [-b [BLOCK]] [--verbose]`

## Setup 
- Clone the repository `git clone https://github.com/zstoebs/CSfMRI-TS.git`.
- Install dependencies
	1. Install Anaconda. Create a new environment with the dependencies: `conda env create -f no-builds.yml` (*suggested*), or
	2. Install dependencies with `pip -r requirements.txt`.

## References
[1] E. J. Cand`es et al., “Compressive sampling,” in Proceedings of the international congress of mathematicians, vol. 3, pp. 1433–1452, Citeseer, 2006.
[2] D. Angelosante, G. B. Giannakis, and E. Grossi, “Compressed sensing of time-varying signals,” in 2009 16th International Conference on Digital Signal Processing, pp. 1–8, IEEE, 2009.
[3] X. Zong, J. Lee, A. J. Poplawsky, S.-G. Kim, and J. C. Ye, “Compressed sensing fmri using gradient-recalled echo and epi sequences,” NeuroImage, vol. 92, pp. 312–321, 2014.
[4] O. Jeromin, M. S. Pattichis, and V. D. Calhoun, “Optimal compressed sensing reconstructions of fmri using 2d deterministic and stochastic sampling geometries,” Biomedical engineering online, vol. 11, no. 1, pp. 1–36, 2012.
[5] A. Domahidi, E. Chu, and S. Boyd, “ECOS: An SOCP solver for embedded systems,” in European Control Conference (ECC), pp. 3071–3076, 2013.
[6] G. Andrew and J. Gao, “Scalable training of l 1-regularized log-linear models,” in Proceedings of the 24th international conference on Machine learning, pp. 33–40, 2007.
[7] Z. Zhang and B. D. Rao, “Extension of sbl algorithms for the recovery of block sparse signals with intra-block correlation,” IEEE Transactions on Signal Processing, vol. 61, no. 8, pp. 2009– 2015, 2013.
[8] P. Wolfe, “Convergence conditions for ascent methods,” SIAM Review, vol. 11, no. 2, pp. 226– 235, 1969.
[9] H. Park and X. Liu, “Study on compressed sensing of action potential,” arXiv preprint arXiv:2102.00284, 2021.
[10] A. Jalal, M. Arvinte, G. Daras, E. Price, A. G. Dimakis, and J. Tamir, “Robust compressed sensing mri with deep generative priors,” Advances in Neural Information Processing Systems, vol. 34, pp. 14938–14954, 2021.
[11] X. Li, T. Cao, Y. Tong, X. Ma, Z. Niu, and H. Guo, “Deep residual network for highly accel- erated fmri reconstruction using variable density spiral trajectory,” Neurocomputing, vol. 398, pp. 338–346, 2020.
