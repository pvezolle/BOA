# -*- coding: utf-8 -*-

"""
#====================================================================
 Date: January 2021,  Pascal Vezolle

 Bayesian Optimization packages comparison using bo_bench class

#====================================================================
"""

import numpy as np
import os,sys
from common.boa_examples_utils import BoaExamplesUtils
from boaas_sdk import BOaaSClient
from math import sqrt,sin
from bo_bench import bo_bench
import time

# ======================================================
# OBJECTIVE FUNCTION DEFINITION
# ======================================================

# Some examples for testing
# ======================================================
# **** Schwefel function
Ndims= 1
domains=[-500,500,Ndims]
fsolutions=-1.*Ndims*418.9829
def obj_funcs(x):
   """
     Schwefel function
          f(x)=f(x1,x2,...,xn) = 418.9829 * n - (sum(x .* sin(sqrt(abs(x))), 2))
   or
     f(x)=f(x1,x2,...,xn) =  - (sum(x .* sin(sqrt(abs(x))), 2))

     f(x*)=0 at x*=(420.9687,...,20.9687)
   """
   #y = 418.9829*Ndim
   y = 0.
   for i in range(Ndims):
      y += -1.0*(x[i]*sin(sqrt(abs(x[i]))))
   return y

# Euclidian distance in a hypercube benchmark
#size0=1600.
#size1=4800.
size0=-1.
size1=1.
Ndimh=6
domainh=[size0,size1,Ndimh]
fsolutionh=-1*sqrt(Ndimh*(size1-size0)*(size1-size0))
def obj_funch(x):
   """
      Euclidian distance in a hypercube benchmark
      f(x*)=-sqrt(sum(size0-size1)**2)),  x*=[size1,...,size1]
      
   """
   Ndim=6          # hypercube dimension = number of parameter
   x0_corner=np.zeros(Ndimh,dtype=float)     # original point size0 for the euclidian distance
   x0_corner[:]=size0
   y = -1 * np.linalg.norm(x0_corner-x)
   return y

# camelback function
domainc=[-2.,2.,2]
fsolutionc=-1.0316
def obj_funcc(x):
   """
        the six humped camelback function. This function has global minima
                f(x) = -1.0316 at x = (0.0898, -0.7126) and (-0.0898, 0.7126))
   """
   y = ((4 -2.1*(x[0]*x[0]) + (x[0]*x[0]*x[0]*x[0]) / 3.0)*(x[0]*x[0]) + x[0] * x[1] + (-4 + 4*(x[1]*x[1])) * (x[1]*x[1]))
   return y


# EXECUTION
print(" ** enter in BO bench ** \n")

# BOA server login config
"""
  BOA parameter: --hostname localhost --port 8443 --epochs 40
                     -ho 129.40.42.16 -e 40
"""
example_description = """
       This code provides an example of using bounds optimization with BOA. Bounds is
        useful when our domain is too large to fit into memory, or if we do not know
        what the resolution of our grid should be across one or more of its dimensions.
      """

args = BoaExamplesUtils.parse_commandline_args(example_description,default_epochs=10)
hostname = BoaExamplesUtils.get_connection_url(args)
ntrials=args.epochs

#hostname="https://loclahost:8443"
#ntrials=10

# Benchmark plan
#==============================================
# 2x times each runs
#   3x optimizer=basinhopping,direct,cobyla
#     1x adaptive_expected_improvement
#       2x use_scikit: Flase and True
#         2x seed:None and 2021
#     1x expected_improvement
#       2x epsilon 0.1 and 0.01
#          2x use_scikit: Flase and True
#            2x seed:None and 2001
#total number of runs= 2 x (4 + 8)         
#=============================================

#test=["boa","dragonfly"] # list of package to test
test=["boa"] # list of package to test

obj_func=obj_funch     # objective function hypercube
fsolution=fsolutionh   # objective function solution [optional]
domain=domainh         
dtype=1                

#bench=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest="Hyper6D_basinh_norm_patched",host=hostname,
#                  optimizer="basinhopping",verbose=True)
#bench.plot_graph()        #  execute and plot best values

boa_name_domain=[{"name":"x1"},{"name":"x1"},{"name":"x3"},{"name":"x4"},{"name":"x5"},{"name":"x6"}]
boa_domain=[[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.],[-1.,1.]]
#boa_domain=[[1600.,4800],[1600.,4800],[1600.,4800],[1600.,4800],[1600.,4800],[1600.,4800]]

#config item: [nametest,optimizer,use_scikit,seed,acq,epsilon]
# Iteration loop
run_ok=True
#optimizer="direct"
#optimizer="cobyla"
optimizer="basinhopping"
opt="basin"
for ietr in range(1):
  print(" *** Iteration *** ")
  config=[]
  config.append(["PASCALHyp6D_{0}_aei_it{1}".format(opt,ietr),optimizer,False,None,"adaptive_expected_improvement",0.1])
  config.append(["Hyp6D_{0}_aei_seed_it{1}".format(opt,ietr),optimizer,False,2021,"adaptive_expected_improvement",0.1])
  config.append(["Hyp6D_{0}_aei_sci_it{1}".format(opt,ietr),optimizer,True,None,"adaptive_expected_improvement",0.1])
  config.append(["Hyp6D_{0}_aei_sci_seed_it{1}".format(opt,ietr),optimizer,True,2021,"adaptive_expected_improvement",0.1])

  config.append(["Hyp6D_{0}_ei_01_it{1}".format(opt,ietr),optimizer,False,None,"expected_improvement",0.1])
  config.append(["Hyp6D_{0}_ei_01_seed_it{1}".format(opt,ietr),optimizer,False,2021,"expected_improvement",0.1])
  config.append(["Hyp6D_{0}_ei_01_sci_it{1}".format(opt,ietr),optimizer,True,None,"expected_improvement",0.1])
  config.append(["Hyp6D_{0}_ei_01_sci_seed_it{1}".format(opt,ietr),optimizer,True,2021,"expected_improvement",0.1])

  config.append(["Hyp6D_{0}_ei_001_it{1}".format(opt,ietr),optimizer,False,None,"expected_improvement",0.01])
  config.append(["Hyp6D_{0}_ei_001_seed_it{1}".format(opt,ietr),optimizer,False,2021,"expected_improvement",0.01])
  config.append(["Hyp6D_{0}_ei_001_sci_it{1}".format(opt,ietr),optimizer,True,None,"expected_improvement",0.01])
  config.append(["Hyp6D_{0}_ei_001_sci_seed_it{1}".format(opt,ietr),optimizer,True,2021,"expected_improvement",0.01])

  #for i in range(len(config)):
  for i in range(1):
    experiment_config = {
        "name": config[i][0], "domain": boa_name_domain,
        "model":{"gaussian_process": {
           "kernel_func": "Matern52", "scale_y": False, "scale_x": False, "noise_kernel": False, "use_scikit": config[i][2] }},
        "optimization_type": "min",
        "optimizer": config[i][1],
        "initialization": {
          "type": "random", "random": { "no_samples": 5, "seed": config[i][3] } },
        "sampling_function": {
        "type": config[i][4], "epsilon": config[i][5], "optimize_acq": True, "outlier": False, "bounds": boa_domain }
     }
    print(" \n config \n",experiment_config)
    if run_ok:
      bench=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest=config[i][0],host=hostname,
                experiment_config=experiment_config ,verbose=True)
      bench.plot_graph()        #  execute and plot best values
      bench.plot_param()        # plot parameter graphes
      cc=bench.getconfig()
      nametest=config[i][0]+".txt" 
      rr=bench.getresult(namefile=nametest)
      del bench

print(" ===================================== ")

