from os import mkdir
import pickle
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier

class gplearn_regressor:
    def __init__(self, 
        tournament_size=20,
        init_depth=(1, 4),
        init_method='half and half',
        metric='mean absolute error',
        parsimony_coefficient=0.00001,
        p_crossover=0.9,
        p_subtree_mutation=0.01, 
        p_hoist_mutation=0.01, 
        p_point_mutation=0.01, 
        p_point_replace=0.05,
        max_samples=1.0,
        function_set= ('add', 'sub', 'mul', 'div', 'log','sqrt'),
        pop_size=1000,
        ngen=500):
        self.est = SymbolicClassifier(
                        tournament_size=tournament_size,
                        init_depth=init_depth,
                        init_method='half and half',
                        parsimony_coefficient=parsimony_coefficient,
                        p_crossover=0.9,
                        p_subtree_mutation=0.01, 
                        p_hoist_mutation=0.01, 
                        p_point_mutation=0.01, 
                        p_point_replace=0.05,
                        max_samples=1.0,
                        function_set= function_set,
                        population_size=pop_size,
                        generations=ngen
                       )
    def fit(self,X_train, y_train):
        self.est.fit(X_train, y_train)
    def predict(self,X_test, transform_to_binary=True):
        y_pred = self.est.predict(X_test)
        if transform_to_binary:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            return y_pred 
        return y_pred 
    def score(self, X_test, y_test):
        return self.est.score(X_test, y_test)

    
def save_gplearn_model_pkl(population_size,
    n_gen, 
    seed,
    pars_coeff,
    estimator,
    dataset_type = 'loan_type',
    folder_path = '/Users/paolovincenzofreieslebendeblasio/finpack/models/symbolic_regression/trained_symbolic_models/gplearn'):
    
    if dataset_type == 'loan_type':
        datapath = folder_path +  '/loan_type/seed_{}/'.format(seed)
    elif dataset_type == 'company_bankruptcy':
        datapath = folder_path + '/company_bankruptcy/seed_{}/'.format(seed)
    try: 
        mkdir(datapath)
    except Exception as e:
        print(e, "ERROR")
        pass
    complete_save_path = datapath + 'gplearn_pop_sz_' + str(population_size) + '_' + 'n_gen_' + str(n_gen) + '_pars_coeff_' + str(pars_coeff) + '.pkl'
    with open(complete_save_path, 'wb') as f:
        pickle.dump(estimator, f)

def get_params_from_best_model(hyper_params):
    population_size = hyper_params['population_size']
    n_gen = hyper_params['generations']
    pars_coeff = hyper_params['parsimony_coefficient']
    seed = hyper_params['random_state']
    return population_size, n_gen, pars_coeff, seed

def load_gplearn_model(population_size,n_gen, seed, datapath):
    complete_save_path = datapath + 'gplearn_pop_sz_' + str(population_size) + '_' + 'n_gen_' +str(n_gen) + '_seed_'+ str(seed) + '.pkl'
    with open(complete_save_path, 'rb') as f:
        est = pickle.load(f)
    return est 
    
