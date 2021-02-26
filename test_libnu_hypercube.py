
##Test only the numerical library

import numpy as np
from bayesian_opt.sampling_functions.adaptive_ei_sampler import AdaptiveEISampler as egs
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from gp_pro.gp_base import BasicGP as GP
from gp_pro.kernels.kernel_matern import KernelMatern as Matern1
from gp_pro.kernels.kernel_rbf import KernelRBF as RBF
from math import sqrt,sin
import gp_pro



param=[]
value=[]
# Euclidian distance in a hypercube benchmark
#size0=1600.
#size1=4800.
size0=-1.
size1=1.
Ndimh=6
domainh=[size0,size1,Ndimh]
fsolutionh=-1*sqrt(Ndimh*(size1-size0)*(size1-size0))
def hypercube(x):
   """
      Euclidian distance in a hypercube benchmark
      f(x*)=-sqrt(sum(size0-size1)**2)),  x*=[size1,...,size1]
      
   """
   global param,value
   Ndim=6          # hypercube dimension = number of parameter
   x0_corner=np.zeros(Ndimh,dtype=float)     # original point size0 for the euclidian distance
   x0_corner[:]=size0
   y = -1 * np.linalg.norm(x0_corner-x)
   param.append(x)
   value.append(y)
   print(" \n **** new parameter ",x," value: ",y," \n")
   return y


# gp_pro
kern = Matern1(nu=2.5, length_scale=1.0)
gp = GP(kernel=kern)

# scilearn
#kern = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=1.0)
#kern = Matern(nu=2.5, length_scale=1.0) 
#gp = gpr(kernel=kern) 

print("Init")
sampler = egs(gp, hypercube, 10, bounds=[(-1.,1.),(-1.,1.),(-1.,1.),(-1.,1.),(-1.,1.),(-1.,1.)], optimize_acq=True,
            optimizer='basinhopping',otype='min', epsilon=0.3)
            #optimizer='direct',otype='min', epsilon=0.3)

print("Running ")
sampler.run(epochs=10, print_level=2)

print(" \n ************ Solution ")
for i in range(len(param)):
    print(" epoch: {0}, parameter: {1}, value:{2} ".format(i,param[i],value[i]))
