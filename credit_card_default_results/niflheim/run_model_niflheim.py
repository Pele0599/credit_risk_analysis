import sys
import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from read_file import read_file,transform_dataset_features_labels
import pdb
import numpy as np
import json
import os
import inspect
from utils import jsonify, split_dataset
from symbolic_utils import get_sym_model

def evaluate_model(dataset, results_path, random_state, est_name, est, 
                   hyper_params, complexity, model, test=False, 
                   target_noise=0.0, feature_noise=0.0, 
                   n_samples=10000, scale_x = True,scale_y = True,
                   split_data=None,data_subset_size=2000,weighted_sampling=False,data_splits=10,
                   pre_train=None, skip_tuning=False, sym_data=False):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)
    if hasattr(est, 'random_state'):
        est.random_state = random_state

    ##################################################
    # setup data
    ##################################################
    
    dataset_name = dataset.split('/')[-1][:-7]    
    #If we split data into multiple subdatasets, then we want 
    # a json containing the results from all the different datasets 

    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)


    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scale_y:
        print('scaling y')
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train

    # add noise to the target
    

    # run any method-specific pre_train routines

    ################################################## 
    # define CV strategy for hyperparam tuning
    ################################################## 
    # define a test mode with fewer splits, no hyper_params, and few iterations

    n_splits = 5


    cv = KFold(n_splits=n_splits, shuffle=True,random_state=random_state)

    grid_est = HalvingGridSearchCV(est,cv=cv, param_grid=hyper_params,
            verbose=2, n_jobs=1, scoring='r2', error_score=0.0)

    ################################################## 
    # Fit models
    ################################################## 
    print('training',grid_est)
    t0p = time.process_time()
    t0t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_est.fit(X_train_scaled, y_train_scaled)
    process_time = time.process_time() - t0p
    time_time = time.time() - t0t
    print('Training time measures:',process_time, time_time)
    best_est = grid_est.best_estimator_
    
    # Once we have the best estimator, based on hyperparameter opt
    # we want to find the consistency of a given expresion. 
    
    ##################################################
    # store results
    ##################################################

    dataset_name = dataset.split('/')[-1][:-7]
    results = {
        'dataset':dataset_name,
        'algorithm':est_name,
        'params':jsonify(best_est.get_params()),
        'random_state':random_state,
        'process_time': process_time, 
        'time_time': time_time, 
        'target_noise': target_noise,
        'feature_noise': feature_noise,
    }
    if sym_data:
        results['true_model'] = true_model

    # get the size of the final model
    if complexity == None:
        results['model_size'] = int(features.shape[1])
    else:
        results['model_size'] = int(complexity(best_est))

    # get the final symbolic model as a string
    if model == None:
        results['symbolic_model'] = 'not implemented'
    else:
        if 'X' in inspect.signature(model).parameters.keys():
            results['symbolic_model'] = model(best_est, X_train_scaled)
        else:
            results['symbolic_model'] = model(best_est)

    # scores
    pred = grid_est.predict

    for fold, target, X in zip(['train','test'],
                                [y_train, y_test], 
                                [X_train_scaled, X_test_scaled]
                                ):
        for score, scorer in [('mse',mean_squared_error),
                                ('mae',mean_absolute_error),
                                ('r2', r2_score)
                                ]:
            y_pred = np.asarray(pred(X)).reshape(-1,1)
            if scale_y:
                y_pred = sc_y.inverse_transform(y_pred)
            results[score + '_' + fold] = scorer(target, y_pred) 
    
    ##################################################
    # write to file
    ##################################################
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = (results_path + '/' + dataset_name + '_' + est_name + '_' 
                    + str(random_state))
    if target_noise > 0:
        save_file += '_target-noise'+str(target_noise)
    if feature_noise > 0:
        save_file += '_feature-noise'+str(feature_noise)

    print('save_file:',save_file)

    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(results), out, indent=4)

# store CV detailed results
# turning off for now as I dont think we'll need this for our analysis
# cv_results = grid_est.cv_results_
# cv_results['random_state'] = random_state

# with open(save_file + '_cv_results.json', 'w') as out:
#     json.dump(jsonify(cv_results), out, indent=4)

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

# parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results_test', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                        help='Used for testing a minimal version')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-sym_data',action='store_true', dest='SYM_DATA', 
                        help='Use symbolic dataset settings')
    parser.add_argument('-skip_tuning',action='store_true', dest='SKIP_TUNE', 
                        default=False, help='Dont tune the estimator')
    parser.add_argument('-data_splits',action='store_true', dest='SPLIT_DATA', 
                        default=False, help='Split data into subsets, and train on these')

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,
                                        globals(),
                                        locals(),
                                        ['*']
                                    )

    print('algorithm:',algorithm.est)
    if 'hyper_params' not in dir(algorithm):
        algorithm.hyper_params = {}
    print('hyperparams:',algorithm.hyper_params)

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    # check for conflicts btw cmd line args and eval_kwargs
    if args.SYM_DATA:
        eval_kwargs['scale_x'] = False
        eval_kwargs['scale_y'] = False
        eval_kwargs['skip_tuning'] = True
        eval_kwargs['sym_data'] = True
    if args.SKIP_TUNE:
        eval_kwargs['skip_tuning'] = True
    print(args.RANDOM_STATE,'RANDOM STATE')
    evaluate_model(args.INPUT_FILE, args.RDIR, args.RANDOM_STATE, args.ALG,
                    algorithm.est, algorithm.hyper_params, algorithm.complexity,
                    algorithm.model, test = args.TEST, 
                    split_data = args.SPLIT_DATA,
                    data_subset_size=1000,weighted_sampling=True,data_splits=3,
                    target_noise=args.Y_NOISE, feature_noise=args.X_NOISE,
                    **eval_kwargs)
