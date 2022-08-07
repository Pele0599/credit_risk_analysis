from pickle import POP
from gplearn.genetic import SymbolicClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
from data.datapaths import datapaths
from read_data import *
from models.symbolic_regression.gplearn import save_gplearn_model_pkl, get_params_from_best_model
hyper_params = []
pars_coeff_init = 0.0001
pars_coeffs = []#[0.01, 0.001, 0.0001, 0.00001]
random_state_data  = 42

random_state_models = 42
function_set = ('add', 'sub', 'mul', 'div', 'log','sqrt')

populations = [1500,3000,5000]
generations = [5000,3000,2000]
for p, g in zip(populations, generations):#zip([1000,3000,5000],[5000,3000,2000]):
    hyper_params.append({
        'population_size' : [p],
        'generations' : [g],
        'function_set': [function_set],
        'parsimony_coefficient' : [pars_coeff_init],
        'random_state' : [random_state_models], 
        })

est = SymbolicClassifier(
        tournament_size=20,
        init_depth=(1, 8),
        init_method='half and half',
        parsimony_coefficient=0.001,
        p_crossover=0.5,
        p_subtree_mutation=0.01, 
        p_hoist_mutation=0.01, 
        p_point_mutation=0.01, 
        p_point_replace=0.05,
        max_samples=1.0,
        function_set= ('add', 'sub', 'mul', 'div', 'log',
                        'sqrt', 'sin','cos'),
        population_size=1000,
        generations=500
        )

datapath = datapaths['credit_card_acceptance']
X,y = get_credit_card_data_single_output(datapath)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)

n_splits = 5

cv =  StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=random_state_data)
grid_est = HalvingGridSearchCV(est,cv=cv, param_grid=hyper_params,
        verbose=0, n_jobs=1, )
grid_est.fit(X_train, y_train)

# print('--------------')
# print(grid_est.best_params_, 'Best estimator parameters')
# print(grid_est.best_estimator_, 'Best estimator')
# print('--------------')

POPULATION_SIZE, N_GEN, PARS_COEFF, SEED = get_params_from_best_model(grid_est.best_params_,)
save_gplearn_model_pkl(POPULATION_SIZE, N_GEN, SEED, PARS_COEFF, 
    grid_est.best_estimator_,
    dataset_type='loan_type',
    folder_path='/home/energy/pvifr/cred_anal/credit_risk_analysis/credit_card_default_results/trained_models')

for pars_coeff in pars_coeffs:
    for hyper_param in hyper_params:
        hyper_param['parsimony_coefficient'] = [pars_coeff]
    
    cv =  StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=random_state_data)
    grid_est = HalvingGridSearchCV(est,cv=cv, param_grid=hyper_params,
        verbose=0, n_jobs=1, )
    grid_est.fit(X_train, y_train)
    POPULATION_SIZE, N_GEN, PARS_COEFF, SEED = get_params_from_best_model(grid_est.best_params_,)

    save_gplearn_model_pkl(POPULATION_SIZE, N_GEN, SEED, PARS_COEFF,
        grid_est.best_estimator_,
        dataset_type='loan_type',
	folder_path='/home/energy/pvifr/cred_anal/credit_risk_analysis/credit_card_default_results/trained_models')
    


