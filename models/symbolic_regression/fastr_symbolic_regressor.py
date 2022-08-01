from fastsr.estimators.symbolic_regression import SymbolicRegression

def get_pkl_file_name_fastsr(population_size,seed, ):
    datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/models/symbolic_regression/trained_symbolic_models/fastsr/'
    return datapath + 'pop_sz_' + str(population_size) + '_' + 'seed_'+ str(seed)
    
    
