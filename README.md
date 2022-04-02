# CSfMRI-TS
Author: Zach Stoebner

EECE 8396 S22


## Description
Exploring compressed sensing fMRI time series.

## Parts
- [tutorial](/tutorial/): background tutorial for compressed sensing with examples for:
	1. the traditional convex optimization approach to recover a wave signal + a downsized 2D image from the discrete cosine transform, and 
	2. an L-BFGS gradient descent approach to recover a full-sized 2D image from the discrete cosine transform. 
- [opt](/optCSfMRI-TS.py): L1 minimization of voxel time series through general convex optimization formulation. So far:
	1. Analysis of a task-based fMRI to identify the most active voxel correlated to the task and exploring sparse recovery related to the voxel activation, in the context of the Nyquist rates defined by the HRF and task response function. 

## Usage
- Clone the repository `git clone https://github.com/zstoebs/CSfMRI-TS.git`.
- Install dependencies
	1. Install Anaconda. Create a new environment with the dependencies: `conda env create -f environment.yml` (*suggested*), or
	2. Install dependencies with `pip -r requirements.txt`.
