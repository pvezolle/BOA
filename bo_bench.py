# -*- coding: utf-8 -*-

"""
#====================================================================
 Date: January 2021,  Pascal Vezolle

 Bayesian Optimization packages comparison 
 space design: bounds domain
 batch: Parallel=1 <=> 1 new parameter set per epoch
 function: minimize
 type: python class (bo_bench)

 packages:
         - scikit-optimize - https://scikit-optimize.github.io/stable/
              name : skopt
              sampler :  https://scikit-optimize.github.io/stable/modules/minimize_functions.html#skopt.dummy_minimize
                        - Gaussian Process (gp_minimize), default with Matern Kernel, 10 init points, gp_edge acq_func
                        - random (dummy_minimize)
                        - decision tree (forest_minimier)
                        - gradient boosting (gbrt_minimize)
         - Optuna - https://optuna.org/ 
              name : optuna
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nondominated Sorting Genetic Algorithm II"
                       can pruners automatically stop unpromising trials 
         - Hyperopt 
              name : hyperopt
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II"
              sampler: TPE, Random, Adaptive TPE (default) (require lightgbm package installed)
         - Dragonfly - https://dragonfly-opt.readthedocs.io/en/master/
              name : dragonfly
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II"
              sampler: bayesian, random, direct (dividing rectangles), PDOO
         - BOA, bounds
              name : boa
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II"
              sampler : Gaussian Process, Matern52 kernel, 10 init points

 comment

#====================================================================
"""

import optuna
import hyperopt
#from hyperopt import hp,fmin,tpe,rand,atpe
from hyperopt import hp,fmin,tpe,rand
from skopt.space import Real, Integer
from skopt import gp_minimize,Optimizer,dummy_minimize, gbrt_minimize, forest_minimize
from dragonfly import maximise_function,minimise_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os,sys
from common.boa_examples_utils import BoaExamplesUtils
from boaas_sdk import BOaaSClient
from hyperopt import Trials
from math import sqrt,sin
import pprint
import time

class bo_bench:
  def __init__(self,func,dtype,domain,fsolution=-100000.,packages=[],ntrials=10,nametest="BObench",
               host="",user={},experiment_config={},optimizer="basinhopping",verbose=False):
    """ 
       Bayesian Optimization packages comparison 
       space design: bounds domain
       batch: Parallel=1 <=> 1 new parameter set per epoch
       function: minimize
       type: python class (bo_bench)

       packages:
         - scikit-optimize - https://scikit-optimize.github.io/stable/
              name : skopt
              sampler :  https://scikit-optimize.github.io/stable/modules/minimize_functions.html#skopt.dummy_minimize
                        - Gaussian Process (gp_minimize), default with Matern Kernel, 10 init points, gp_edge acq_func
                        - random (dummy_minimize)
                        - decision tree (forest_minimier)
                        - gradient boosting (gbrt_minimize)
         - Optuna - https://optuna.org/ 
              name : optuna
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nondominated Sorting Genetic Algorithm II"
                       can pruners automatically stop unpromising trials 
         - Hyperopt 
              name : hyperopt
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II"
              sampler: TPE, Random, Adaptive TPE (default) (require lightgbm package installed)
         - Dragonfly - https://dragonfly-opt.readthedocs.io/en/master/
              name : dragonfly
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II"
              sampler: bayesian, random, direct (dividing rectangles), PDOO
         - BOA, bounds

      Parameters:
        func       : objective function, f(x)
        fsolution  : min value of the objective function, only used in the plot graphes (real)
        dtype      : format of space domain (integer)
                     1: [size0,size1,Ndim] same bounds for the Ndim parameters
                     2: [[x1,y1],..,[xn,yn]] specific bounds for every Ndim parameters (implementation in progress)
        domain     : space domain dtpye 1 or 2 (python list)
        packages   : list of packagase to compare  (python liston)
                     options: ["skopt","optuna","hyperopt","dragonfly","boa"]
        ntrials    : number of trails/epoches  (integer)
        nametest   : name of the test, used to name the plot files (str)
        host       : Boa servier address + port (str)
        user       : BOA user (dict)
        experiment_config : BOA experiment_config, if None default: GP, Martern52, 10 init point
        optimizer  : BOA optimizer for bouns domain, options: "direct", "cobyla", "basinhopping" 
        verbose    : for debug
    """
    config={}
    self.obj_func=func   
    self.fsolution = fsolution
    self.ntrials=ntrials  
    self.verbose=verbose
    config["fsolution"]=fsolution
    config["ntrials"]=ntrials
    config["verbose"]=verbose

    # set packages to execute
    self.run_skopt=False
    self.run_optuna=False
    self.run_hyperopt=False
    self.run_dragonfly=False
    self.run_boa=False
    if "skopt" in packages:
      self.run_skopt=True
    if "hyperopt" in packages:
      self.run_hyperopt=True
    if "optuna" in packages:
      self.run_optuna=True
    if "dragonfly" in packages:
      self.run_dragonfly=True
    if "boa" in packages:
      self.run_boa=True
    config["run_skopt"]=self.run_skopt
    config["run_optuna"]=self.run_optuna
    config["run_hyperopt"]=self.run_hyperopt
    config["run_dragonfly"]=self.run_dragonfly
    config["run_boa"]=self.run_boa

    # build space bounds domains
    self.boa_name_domain=list()  
    self.boa_domain=[]          
    self.skopt_domain=[]       
    self.dragonfly_domain=[]
    self.hyperopt_domain=[]

    if dtype == 1:
      if len(domain) != 3:
        raise ValueError(" bo_bench class, error in domain definition")
      self.size0 = domain[0]
      self.size1 = domain[1]
      self.Ndim = domain[2]
      self.buildomain1()
    elif dtype ==2:
        raise ValueError(" bo_bench class, domain dtype 2 not yet implemented")
    else:
        raise ValueError(" bo_bench class, error wrong domain dtype")

    config["dtype"]=1
    config["domain"]=domain
    config["boa_name_domain"]=self.boa_name_domain
    config["boa_domain"]=self.boa_domain
    config["skopt_domain"]=self.skopt_domain
    config["dragonfly_domain"]=self.dragonfly_domain
    config["hyperopt_domain"]=self.hyperopt_domain

    # graph configuration
    self.hh=time.strftime("%c", time.gmtime())
    hh1=time.strftime("%d%b%G_%H%M",time.gmtime())
    self.nametest=nametest # name of the test
    self.namebest=nametest+"_best_{}.png".format(hh1)      # name file of best result graph
    self.nameparam=nametest+"_param_{}.png".format(hh1)    # name file of parameter graphes
    config["nametest"]=self.nametest
    config["namebest"]=self.namebest
    config["nameparam"]=self.nameparam

    # BOA CONFIG
    self.optimizer=optimizer
    if len(host) != 0:
      self.host=host
    else:
      self.host='http://129.40.42.16'

    if bool(user):
      self.user=user
    else:
      self.user = {"_id": "boa_test@test.com", "name": "boa",
                   "password": "password", "confirm_password": "password" }

    if bool(experiment_config):
       self.experiment_config = experiment_config
    else:
       self.experiment_config = self.boa_default_experiment_config()
   
    if self.verbose:
      print(" BOA experiment_config ",self.experiment_config)

    config["BOA_optimizer"]=self.optimizer
    config["host"]=self.host
    self.config=config

    # internal variables
    self.niter=0
    self.niterboa=0
    self.niterdragonfly=0

    self.skopt_time=0.
    self.optuna_time=0.
    self.hyperopt_time=0.
    self.dragonfly_time=0.
    self.boa_time=0.

    # list to save execution parameters
    # *********************************
    self.skopt_param=[]
    self.skopt_func=[]
    self.skopt_epoches=[]
    self.skopt_values=[]
    self.skopt_func=[]
    self.optuna_param=[]
    self.optuna_func=[]
    self.optuna_epoches=[]
    self.optuna_values=[]
    self.hyperopt_param=[]
    self.hyperopt_func=[]
    self.hyperopt_epoches=[]
    self.hyperopt_values=[]
    self.dragonfly_param=[]
    self.dragonfly_func=[]
    self.dragonfly_epoches=[]
    self.dragonfly_values=[]
    self.boa_param=[]
    self.boa_func=[]
    self.boa_epoches=[]
    self.boa_values=[]

  # best epoch number and values
    self.skopt_bestepoch=int(1000000)
    self.skopt_bestvalue=0.
    self.optuna_bestepoch=int(1000000)
    self.optuna_bestvalue=0.
    self.hyperopt_bestepoch=int(100000)
    self.hyperopt_bestvalue=0.
    self.dragonfly_bestepoch=int(100000)
    self.dragonfly_bestvalue=0.
    self.boa_bestepoch=int(100000)
    self.boa_bestvalue=0.

  # ==== end __init__
  #======================================================

  def getconfig(self):
    """ 
       print in stdout configuration parameters
    """
    print(" \n ****  BO Bench configuration ")
    for keys,values in self.config.items():
       print("{0:18s} : {1} ".format(keys,values))
    print(" \n *******************************")

    return self.config

  def getresult(self,namefile=""):
    """
        give test results in a dict, d["name_param"],d["name_value"],d["name_time"] for name in ["skopt","optuna","hyperopt","dragonfly","boa"]
        results can be save in the file, need to provide namefile
    """
    print(" \n ****  BO Bench getresult ")
    results={}
    results["skopt_param"]=self.skopt_param
    results["skopt_value"]=self.skopt_func
    results["skopt_time"]=self.skopt_time
    results["optuna_param"]=self.optuna_param
    results["optuna_value"]=self.optuna_func 
    results["optuna_time"]=self.optuna_time 
    results["hyperopt_param"]=self.hyperopt_param
    results["hyperopt_value"]=self.hyperopt_func
    results["hyperopt_time"]=self.hyperopt_time
    results["dragonfly_param"]=self.dragonfly_param
    results["dragonfly_value"]=self.dragonfly_func
    results["dragonfly_time"]=self.dragonfly_time
    results["boa_param"]=self.boa_param
    results["boa_value"]=self.boa_func
    results["boa_time"]=self.boa_time

    if len(namefile) != 0:   # same results in a file
      with open(namefile,'w') as f:
        f.write(" === BO bench results - {0} - {1} \n".format(self.nametest,self.hh)) 
        ll=["skopt","optuna","hyperopt","dragonfly","boa"]
        for p in ll:
          name_time=p+"_time"
          f.write(" \n *** {0} execution **** \n   Elapsed Time: {1}s \n ".format(p,results[name_time]))
          name_param=p+"_param"
          name_value=p+"_value"
          nepoch=len(results[name_param])
          for n in range(nepoch):
             f.write("epoch: {0},  value: {1} \n ".format(n,results[name_value][n]))
             f.write("epoch: {0},  param: {1} \n ".format(n,results[name_param][n]))
      
    return results

  def buildomain1(self):
    """
       internal function to build space domains
    """
    size0=self.size0
    size1=self.size1
    size=[size0,size1]
    for i in range(1,self.Ndim+1):
        param_name="x"+str(i)
        #BOA bound domain definition
        j={}
        j["name"]=param_name
        self.boa_name_domain.append(j)
        self.boa_domain.append(size)

        #Dragonfly domain definition
        self.dragonfly_domain.append(size)

        #Skopt domain definition
        d="Real({0},{1},name='{2}')".format(size0,size1,param_name) 
        self.skopt_domain.append(Real(size0,size1,name='param_name'))

        #Hyperopt domain definition
        self.hyperopt_domain.append(hp.uniform(param_name,size0,size1))
     
    if self.verbose:
       print(" boa_name_domain ",self.boa_name_domain)
       print(" boa_domain ",self.boa_domain)
       print(" skopt_domain ",self.skopt_domain)
       print(" hyperopt_domain ",self.hyperopt_domain)
       print(" dragonfly_domain ",self.dragonfly_domain)

  def boa_default_experiment_config(self):
    """
       internal function to build BO default experiment_config
    """
    experiment_config = {
        "name": self.nametest,

        "domain": self.boa_name_domain,

        "model":{"gaussian_process": {
        "kernel_func": "Matern52",
        "scale_y": True,
        "scale_x": False,
        "noise_kernel": False,
        "use_scikit": False
         }},

        "optimization_type": "min",
        "optimizer": self.optimizer,
        "initialization": {
          "type": "random",
          "random": {
            "no_samples": 10,
            "seed": None
          }
        },
        "sampling_function": {
        "type": "adaptive_expected_improvement",
        "epsilon": 0.03,
        "optimize_acq": True,
        "outlier": False,
        "bounds": self.boa_domain
      }
     }
    return experiment_config

  #+++++++++++++++++++++++++++++++++++++++++++++++++++
  # PACKAGE SECTIONS
  #+++++++++++++++++++++++++++++++++++++++++++++++++++

  #===========================================================
  # Sckit-Optimize
  #===========================================================
  # Define the objective function for skopt
  def objective_skopt(self,x):
    """ 
       internal objective function for scikit-optimize, calls user objective function
    """
    self.niter = self.niter + 1
    self.skopt_param.append(x)
    xR=np.array(x)
    f = self.obj_func(xR)
    self.skopt_func.append(f)
    print(" skopt, epoch: ",self.niter," x: ",x," f(x): ",f)
    return f
    
  # Execute optimization by skopt
  def skopt_exe(self,sampler="gp"):
    """ 
      launch scikit-optimize execution
      parameters:
         gp     : Gaussian Process (gp_minimize), default: with Matern Kernel, 10 init points, gp_edge acq_func
         random : random (dummy_minimize)a
         forest : decision tree (forest_minimier)
         gbrt   : gradient boosting (gbrt_minimize)
      output: 
        lists of epoches and values
    """
    print(" \n ****** Scikit-optimize ***** \n")
    #opt = Optimizer([Real(-2.0, 2.0,name='x1'),Real(-1.0, 1.0,name='x2')],n_initial_points=3)
    #opt = Optimizer(self.skopt_domain)
    #opt.run(self.objective_skopt,n_iter=self.ntrials)
    #res = opt.get_result()

    t0=time.time()
    ntrials=self.ntrials
    objective_skopt = self.objective_skopt
    skopt_domain = self.skopt_domain
    if sampler ==  "gp":
       opt = gp_minimize(objective_skopt,skopt_domain,n_calls=ntrials)
    if sampler ==  "random":
       opt = dummy_minimize(objective_skopt,skopt_domain,n_calls=ntrials)
    if sampler ==  "forest":
       opt = forest_minimize(objective_skopt,skopt_domain,n_calls=ntrials)
    if sampler ==  "gbrt":
       opt = gbrt_minimize(objective_skopt,skopt_domain,n_calls=ntrials)
    t1=time.time()
    self.skopt_time=t1-t0

    res = opt.copy()
    #X=res['x']
    #F=res['fun']
    #print(" result X, F ",X,F)

    best = 100000
    # Update best
    print("\n *** skopt_exe results\n")
    for i in range(ntrials):
        #print(" i \n ",i,res['func_vals'][i])
        if best > res['func_vals'][i]:
            best = res['func_vals'][i]
            self.skopt_bestepoch = i
        self.skopt_epoches.append(i + 1)
        self.skopt_values.append(best)
        print('n:', i, 'vals:', res['x_iters'][i],'result:', res['func_vals'][i], 'best:',best)
    self.skopt_bestvalue = best
    
    return 0


  #===========================================================
  # Optuna
  #===========================================================
  # Define the objective function for Optuna
  def objective_optuna(self,trial):
    """ 
       internal objective function for Optuna, calls user objective function
    """
    x=[]
    for i in range(1,self.Ndim+1):
      param_name="x"+str(i)
      j=trial.suggest_uniform(param_name, self.size0,self.size1)
      x.append(j)
    self.optuna_param.append(x)
    xR=np.array(x)
    f = self.obj_func(xR)
    self.optuna_func.append(f)
    #print(" Optuna, next param: ",x," f(x): ",f)
    return f

  # Execute optimization by Optuna
  def optuna_exe(self,sampler="tpe"):
    """ 
      launch Optuna execution
      parameters:
         tpe    : optuna.samplers.TPESampler
         random : optuna.samplers.RandomSampler()
         cmaes  : optuna.samplers.CmaEsSampler
         nsgaII : optuna.samplers.NSGAIISampler
      output: 
        lists of epoches and values
    """
    print(" \n ****** Optuna ***** \n")

    t0=time.time()
    if sampler == "tpe":
      study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    if sampler == "random":
      study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    if sampler == "cmaes":
      study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
    if sampler == "nsgaII":
      study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler())

    # other options
    #study = optuna.create_study(direction="maximize")
    #study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner())
    #study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    #study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner())

    study.optimize(self.objective_optuna, n_trials=self.ntrials)
    t1=time.time()
    self.optuna_time=t1-t0

    print(study.best_params)
    print(study.best_value)

    best = 100000
    # Update best
    print("\n *** Optuna results\n")
    n=0
    for i in study.trials:
        if best > i.value:
            best = i.value
            self.optuna_bestepoch = i.number
        self.optuna_epoches.append(i.number + 1)
        self.optuna_values.append(best)
        n=n+1
    self.optuna_bestvalue = best

    return 0

  #===========================================================
  # Hyperopt
  #===========================================================
  # Define the objective function for Hyperopt
  def objective_hyperopt(self,args):
    """ 
       internal objective function for hyperopt, calls user objective function
    """
    x=list(args)
    self.hyperopt_param.append(x)
    xR=np.array(x)
    f = self.obj_func(xR)
    self.hyperopt_func.append(f)
    print(" Hyperopt, next param: ",x," f(x): ",f)
    #print(" next param: ",x)
    return f

  # Execute optimization by Hyperopt
  def hyperopt_exe(self,sampler="tpe"):
    """ 
      launch Hyperopt execution
      parameters:
         tpe    : Tree of Parzen Estimators (TPE)
         random : random
         atpe   : Adaptive TPE (default)
      output: 
        lists of epoches and values
    """
    global hyperopt_bestepoch, hyperopt_bestvalue
    print(" \n ****** Hyperopt ***** \n")

    space=self.hyperopt_domain
    ntrials=self.ntrials
    objective_hyperopt = self.objective_hyperopt

    t0=time.time()
    trials = Trials()
    if sampler == "tpe":
      best = fmin(objective_hyperopt, space, algo=tpe.suggest, max_evals=ntrials, trials=trials)
    if sampler == "random":
      best = fmin(objective_hyperopt, space, algo=rand.suggest, max_evals=ntrials, trials=trials)
    #if sampler == "atpe":
    #  best = fmin(objective_hyperopt, space, algo=atpe.suggest, max_evals=ntrials, trials=trials)

    t1=time.time()
    self.hyperopt_time=t1-t0
    #print(best)

    best = 100000
    # Update best
    print("\n *** hyperopt_exe results \n")
    for i, n in zip(trials.trials, range(ntrials)):
        if best > i['result']['loss']:
            best = i['result']['loss']
            self.hyperopt_bestepoch = n
        self.hyperopt_epoches.append(n+1)
        self.hyperopt_values.append(best)
        print('n:', n, 'vals:', i['misc']['vals'], 'result:', i['result'], 'best:',best)
    self.hyperopt_bestvalue = best

    return 0

  #===========================================================
  # Dragonfly
  #===========================================================
  # Define the objective function for Dragonfly
  def objective_drafongly(self,x):
    """ 
       internal objective function for Dragonfly, calls user objective function
    """
    self.niterdragonfly = self.niterdragonfly + 1
    self.dragonfly_param.append(x.tolist())
    f = self.obj_func(x)
    self.dragonfly_func.append(f)
    print(" Dragonfly, epoch: ",self.niterdragonfly," next param: ",x," f(x): ",f)
    return f

  # Execute optimization by Dragonfly
  def dragonfly_exe(self,sampler="bo"):
    """ 
      launch Dragonfly execution
      parameters:
        bo       : Bayesian optimisation
        random   : random
        ea or ga : Evolutionary algorithm (does nor work, need to investigate)
        direct   : Dividing Rectangles
        pdoo     : PDOO.
      output: 
        lists of epoches and values
    """
    print(" \n ****** Dragonfly ***** \n")

    ntrials = self.ntrials
    objective_drafongly = self.objective_drafongly
    dd = self.dragonfly_domain
    t0=time.time()
    if sampler == "bo":
      opt_val, opt_pt, history = minimise_function(objective_drafongly, domain=dd, max_capital=ntrials,opt_method='bo')
    if sampler == "random":
      opt_val, opt_pt, history = minimise_function(objective_drafongly, domain=dd, max_capital=ntrials,opt_method='rand')
    if sampler == "direct":
      opt_val, opt_pt, history = minimise_function(objective_drafongly, domain=dd, max_capital=ntrials,opt_method='direct')
    if sampler == "pdoo":
      opt_val, opt_pt, history = minimise_function(objective_drafongly, domain=dd, max_capital=ntrials,opt_method='pdoo')

    t1=time.time()
    self.dragonfly_time=t1-t0
    print('Optimum Value in %d evals: %0.4f'%(ntrials, opt_val))
    print('Optimum Point: %s'%(opt_pt))

    best = 100000
    # Update best
    print("\n *** Dragonfly_exe results \n")
    for n in range(self.niterdragonfly):
        if best > float(self.dragonfly_func[n]):
            best = float(self.dragonfly_func[n])
            self.dragonfly_bestepoch = n
        self.dragonfly_epoches.append(n+1)
        self.dragonfly_values.append(best)
        print('n:', n, 'results:', self.dragonfly_func[n], 'best:',best)
    self.dragonfly_bestvalue = best

    return 0

  #===========================================================
  # BOA
  #===========================================================
  # Define the objective function for BOA
  def objective_boa(self,x):
    """ 
       internal objective function for BOA, calls user objective function
    """
    self.niterboa = self.niterboa + 1
    self.boa_param.append(x)
    xR=np.array(x)
    f = self.obj_func(xR)
    self.boa_func.append(f)
    print(" BOA, epoch: ",self.niterboa," next param: ",x," f(x): ",f)
    return f

  # Execute optimization by BOA
  def boa_exe(self):
    """
      launch BOA execution
      output: 
        lists of epoches and values
    """
    print(" \n ****** BOA ***** \n")

    host=self.host
    user=self.user
    experiment_config = self.experiment_config
    
    boaas = BOaaSClient(host=host)

    user_login = boaas.login(user)

    t0=time.time()
    print(" BOA user_login",user_login)
    user_token = user_login["logged_in"]["token"]
    print(" BOA user token",user_token)
    create_exp_user_object = { "_id": user["_id"], "token": user_token}
    experiment_res = boaas.create_experiment(create_exp_user_object, experiment_config)
    print(" BOA experiment_res",experiment_res)
    experiment_id = experiment_res["experiment"]["_id"]
    print(" BOA experiment_id",experiment_id)
    boaas.run(experiment_id=experiment_id, user_token=user_token, func=self.objective_boa, no_epochs=self.ntrials, explain=False)
    t1=time.time()
    self.boa_time=t1-t0

    data = boaas.get_experiment_data(experiment_id=experiment_id, user_token=user_token)
    #print("data:")
    #pprint.pprint(data)

    best = 100000
    nbest = 0
    
    obs=data["experiment"]["observations"]
    num_obs = len(obs)
    for i, n in zip(obs, range(num_obs)):
        if i['y'][0] != None:
            if best > i['y'][0]:
                best = i['y'][0]
                nbest = n
        self.boa_epoches.append(n+1)
        self.boa_values.append(best)
        print('n:', n, 'vals:', i['X'], 'result:', i['y'][0], 'best:',best)
    self.boa_bestepoch = nbest
    self.boa_bestvalue = best

    best_observation = boaas.best_observation(experiment_id, user_token)

    print("best observation:",best_observation)
    boaas.stop_experiment(experiment_id=experiment_id, user_token=user_token)

    return 0

  #============================================================================
  #============================================================================
  # launch optimization and plot  functions
  #============================================================================
  #============================================================================

  def plot_graph(self,run=True):
    """
      launch executions and plot best result graph for all packages
      parameter:
        run : True  - launch execution and pot best values graph, (default)
              False - plot best values graph
    """
    graptitle=" "

    # Launch Optimization
    if run:
     if self.run_optuna:
       err = self.optuna_exe()
     sys.stdout.flush()
     if self.run_hyperopt:
       err = self.hyperopt_exe()
     sys.stdout.flush()
     if self.run_boa:
       err = self.boa_exe()
     sys.stdout.flush()
     if self.run_skopt:
       err = self.skopt_exe()
     sys.stdout.flush()
     if self.run_dragonfly:
       rerr = self.dragonfly_exe()
     sys.stdout.flush()

    fig, ax = plt.subplots()
    ax.set_xlabel("trial")
    ax.set_ylabel("f(X)")
    ax.grid()
    if self.run_optuna:
       print(" ee ",self.optuna_epoches)
       print(" values ",self.optuna_values)
       print(" time ",self.optuna_time)
       graptitle =graptitle+" Optuna"
       ax.plot(self.optuna_epoches, self.optuna_values, color="red", label="Optuna\n- Final result:"+f'{self.optuna_values[len(self.optuna_values)-1]:.5f}'+"\n- Elapsed time:"+f'{(self.optuna_time/len(self.optuna_epoches)):.5f}'+"[sec/trial]")
    if self.run_hyperopt:
       graptitle =graptitle+" Hyperopt"
       ax.plot(self.hyperopt_epoches, self.hyperopt_values, color="blue", label="Hyperopt\n- Final result:"+f'{self.hyperopt_values[len(self.hyperopt_values)-1]:.5f}'+"\n- Elapsed time:"+f'{(self.hyperopt_time/len(self.hyperopt_epoches)):.5f}'+"[sec/trial]")
    if self.run_boa:
       graptitle =graptitle+" BOA"
       ax.plot(self.boa_epoches, self.boa_values, color="green", label="IBM BOA\n- Final result:"+f'{self.boa_values[len(self.boa_values)-1]:.5f}'+"\n- Elapsed time:"+f'{(self.boa_time/len(self.boa_epoches)):.5f}'+"[sec/trial]")
    if self.run_skopt:
       graptitle =graptitle+" Scikit-Optimize"
       ax.plot(self.skopt_epoches, self.skopt_values, color="black", label="skopt\n- Final result:"+f'{self.skopt_values[len(self.skopt_values)-1]:.5f}'+"\n- Elapsed time:"+f'{(self.skopt_time/len(self.skopt_epoches)):.5f}'+"[sec/trial]")
    if self.run_dragonfly:
       graptitle =graptitle+" Dragonfly"
       ax.plot(self.dragonfly_epoches, self.dragonfly_values, color="purple", label="Dragonfly\n- Final result:"+f'{self.dragonfly_values[len(self.dragonfly_values)-1]:.5f}'+"\n- Elapsed time:"+f'{(self.dragonfly_time/len(self.dragonfly_epoches)):.5f}'+"[sec/trial]")

    if self.fsolution != -100000.:
       graptitle =graptitle+"\n  {0}, min:{1} \n {2}".format(self.nametest,self.fsolution,self.hh)
    else:
       graptitle =graptitle+"\n  {0}, \n {1}".format(self.nametest,self.hh)
    ax.set_title(graptitle,fontsize=8)

    ax.legend(loc='best',fontsize='x-small')

    plt.savefig(self.namebest, format="png", dpi=300)
    plt.show()


  def plot_param(self):
    """
       plot parameter graph for all packages
    """
    numgraph=0
    if self.run_optuna:
        numgraph = numgraph+1
    if self.run_hyperopt:
        numgraph = numgraph+1
    if self.run_skopt:
        numgraph = numgraph+1
    if self.run_boa:
        numgraph = numgraph+1
    if self.run_dragonfly:
        numgraph = numgraph+1

    if numgraph==0:
        return 0

    ncol = int(sqrt(numgraph))
    nrow = int(numgraph/ncol)
    if ncol*nrow < numgraph:
        nrow = nrow + 1

    plt.rc('xtick', labelsize=6)  
    plt.rc('ytick', labelsize=6)   
    #figure = plt.figure(constrained_layout=True)
    figure = plt.figure()
    plt.suptitle("Parameter values, {0} - {1}".format(self.nametest,self.hh),fontsize=8)

    j=1
    def plot_point(ax,x,y,best,name):
        ax.set_ylabel(name,fontsize='x-small')
        ax.grid()
        for i in range(self.Ndim):
           ax.plot(x,y[i,:], '.b')
           ax.plot(best,y[i,best], '.r')

    if self.run_optuna:
        ll=len(self.optuna_param)
        x=np.linspace(0,ll-1, num=ll, dtype=int)
        y=np.array(self.optuna_param).transpose()
        ax = plt.subplot(nrow,ncol,j)
        plot_point(ax,x,y,best=self.optuna_bestepoch,name=" Optuna ")
        print(" \n Package: Optuna, best value: ",self.optuna_bestvalue,"at epoch ", self.optuna_bestepoch," parameter: ",y[:,self.optuna_bestepoch])
        j=j+1

    if self.run_hyperopt:
        ll=len(self.hyperopt_param)
        x=np.linspace(0,ll-1, num=ll, dtype=int)
        y=np.array(self.hyperopt_param).transpose()
        ax = plt.subplot(nrow,ncol,j)
        plot_point(ax,x,y,best=self.hyperopt_bestepoch,name=" Hyperopt ")
        print(" \n Package: Hyperopt, best value: ",self.hyperopt_bestvalue,"at epoch ", self.hyperopt_bestepoch," parameter: ",y[:,self.hyperopt_bestepoch])
        j=j+1

    if self.run_skopt:
        ll=len(self.skopt_param)
        x=np.linspace(0,ll-1, num=ll, dtype=int)
        y=np.array(self.skopt_param).transpose()
        ax = plt.subplot(nrow,ncol,j)
        plot_point(ax,x,y,best=self.skopt_bestepoch,name=" Scikit-Optmize ")
        print(" \n Package: Scikit-Optimize, best value: ",self.skopt_bestvalue,"at epoch ", self.skopt_bestepoch," parameter: ",y[:,self.skopt_bestepoch])
        j=j+1

    if self.run_boa:
        ll=len(self.boa_param)
        x=np.linspace(0,ll-1, num=ll, dtype=int)
        y=np.array(self.boa_param).transpose()
        ax = plt.subplot(nrow,ncol,j)
        plot_point(ax,x,y,best=self.boa_bestepoch,name=" BOA ")
        print(" \n Package: BOA, best value: ",self.boa_bestvalue,"at epoch ", self.boa_bestepoch," parameter: ",y[:,self.boa_bestepoch])
        j=j+1

    if self.run_dragonfly:
        ll=len(self.dragonfly_param)
        x=np.linspace(0,ll-1, num=ll, dtype=int)
        y=np.array(self.dragonfly_param).transpose()
        ax = plt.subplot(nrow,ncol,j)
        plot_point(ax,x,y,best=self.dragonfly_bestepoch,name=" Dragonfly ")
        print(" \n Package: Dragonfly, best value: ",self.dragonfly_bestvalue,"at epoch ", self.dragonfly_bestepoch," parameter: ",y[:,self.dragonfly_bestepoch])
        j=j+1

    plt.savefig(self.nameparam, format="png", dpi=300)
    plt.show()

  #  === END of bo_bench class ===




if __name__ == '__main__':

 # Some internal examples for testing
 #===============================================================================
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

 #===============================================================================
 # Euclidian distance in a hypercube benchmark
 size0=1600.
 size1=4800.
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

 #===============================================================================
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

 #===============================================================================
 #===============================================================================

 print(" ** enter in BO bench ** \n")

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

 test=["optuna","skopt","hyperopt","boa","dragonfly"]

 domain=domainh
 dtype=1
 fsolution=fsolutionh
 obj_func=obj_funch
   
 bench=bo_bench(obj_func,dtype,domain,fsolution=fsolution,packages=test,ntrials=ntrials,nametest="Hyper6D_basinh",host=hostname,
                  verbose=True)
 print(" type ",type(bench))
 print(" help ",help(bench))
 quit()
 bench.plot_graph()
 bench.plot_param()
 cc=bench.getconfig()
 rr=bench.getresult(namefile="Hyper6D_basinh.txt")
 print(" ===================================== ")

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

