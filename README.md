 #                         Bayesian Optimization packages comparison 
 space design: bounds domain <br />
 batch: Parallel=1 <=> 1 new parameter set per epoch <br />
 function: minimize <br />
 type: python class (bo_bench) <br />
 version: 1.0 <br /> <br />
 
packages:  <br />
        - scikit-optimize - https://scikit-optimize.github.io/stable/  <br />
              name : skopt  <br />
              sampler :  https://scikit-optimize.github.io/stable/modules/minimize_functions.html#skopt.dummy_minimize  <br />
                        - Gaussian Process (gp_minimize), default with Matern Kernel, 10 init points, gp_edge acq_func  <br />
                        - random (dummy_minimize)  <br />
                        - decision tree (forest_minimier)  <br />
                        - gradient boosting (gbrt_minimize)  <br />
        - Optuna - https://optuna.org/  <br />
              name : optuna <br />
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nondominated Sorting Genetic Algorithm II" <br />
                       can pruners automatically stop unpromising trials  <br />
         - Hyperopt  <br />
              name : hyperopt <br />
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II" <br />
              sampler: TPE, Random, Adaptive TPE (default) (require lightgbm package installed) <br />
         - Dragonfly - https://dragonfly-opt.readthedocs.io/en/master/ <br />
              name : dragonfly <br />
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II" <br />
              sampler: bayesian, random, direct (dividing rectangles), PDOO <br />*        - BOA, bounds <br />
              name : boa <br />
              sampler: TPE (default), Random,  CMA-ES algorithm, NSGA-II "Nro ndominated Sorting Genetic Algorithm II" <br />
              sampler : Gaussian Process, Matern52 kernel, 10 init points <br />
 
