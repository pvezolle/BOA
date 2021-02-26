##Test only the numerical library

import numpy as np
from bayesian_opt.sampling_functions.adaptive_ei_sampler import AdaptiveEISampler as egs
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
#from gp_pro.gp_base import GP
#from gp_pro.kernels.kernel_matern import KernelMater as Matern
#from gp_pro.kernels.kernel_rbf import KernelRBF as RBF

def camelback(x):
   y = ((4 -2.1*(x[0]*x[0]) + (x[0]*x[0]*x[0]*x[0]) / 3.0)*(x[0]*x[0]) + x[0] * x[1] + (-4 + 4*(x[1]*x[1])) * (x[1]*x[1]))
   return y

#kern = Matern(nu=2.5, length_scale=1.0)
#gp = GP(kernel=kern)

kern = Matern(nu=2.5, length_scale=1.0)
#gp = gpr(kernel=kern) + WihteKernel(noise_level=1.0)
gp = gpr()

print("setting up")
sampler = egs(gp, camelback, 3, bounds=[(-2.,2.),(-1.,1.)], optimize_acq=True,otype='min', epsilon=0.3)

print(" running ")
sampler.run(epochs=3, print_level=2)
