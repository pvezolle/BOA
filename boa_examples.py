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
size0=1600.
size1=4800.
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
domainc=[-2.,2.,1]
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

# TEST CONFIGURATION


#  bo_bench parameters/options:
#       func       : objective function, f(x)
#       fsolution  : min value of the objective function, only used in the plot graphes (real)
#       dtype      : format of space domain (integer)
#                    1: [size0,size1,Ndim] same bounds for the Ndim parameters
#                    2: [[x1,y1],..,[xn,yn]] specific bounds for every Ndim parameters (implementation in progress)
#       domain     : space domain dtpye 1 or 2 (python list)
#       packages   : list of packagase to compare  (python liston)
#                    options: ["skopt","optuna","hyperopt","dragonfly","boa"]
#       ntrials    : number of trails/epoches  (integer)
#       nametest   : name of the test, used to name the plot files (str)
#       host       : Boa servier address + port (str)
#       user       : BOA user (dict)
#       experiment_config : BOA experiment_config, if None default: GP, Martern52, 10 init point
#       optimizer  : BOA optimizer for bouns domain, options: "direct", "cobyla", "basinhopping" 
#       verbose    : for debug

test=["optuna","skopt","hyperopt","boa","dragonfly"] # list of package to test
obj_func=obj_funch     # objective function
fsolution=fsolutionh   # objective function solution [optional]
domain=domainh         
dtype=1                
   

#bench=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=["skopt"],ntrials=ntrials,nametest="Hyper6D_basinh",host=hostname,
#                  optimizer="basinhopping",verbose=True)
#bench.skopt_exe()          # execute only Scikit-Optmize 
#bench.plot_graph(run=False)  #  plot best values
#bench.plot_param()
#quit()

bench=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest="Hyper6D_basinh",host=hostname,
                  optimizer="basinhopping",verbose=True)
bench.plot_graph()        #  execute and plot best values
bench.plot_param()        # plot parameter graphes
cc=bench.getconfig()
rr=bench.getresult(namefile="Hyper6D_basinh.txt")

print(" ===================================== ")

quit()

bench1=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest="Hyper6D_direct",host=hostname,
                  optimizer="direct",verbose=True)
bench1.plot_graph()
bench1.plot_param()
cc=bench1.getconfig()
rr=bench1.getresult(namefile="Hyper6D_direct.txt")
print(" ===================================== ")

bench2=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest="Hyper6D_cobyla",host=hostname,
                optimizer="cobyla",verbose=True)
bench2.plot_graph()
bench2.plot_param()
cc=bench2.getconfig()
rr=bench2.getresult(namefile="Hyper6D_cobyla.txt")
print(" ===================================== ")

