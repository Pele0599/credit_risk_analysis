from fastsr.estimators.symbolic_regression import SymbolicRegression as fastsr

def get_pkl_file_name_fastsr(population_size,n_gen, seed, ):
    datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/models/symbolic_regression/trained_symbolic_models/fastsr/'
    return datapath + 'pop_sz_' + str(population_size) + '_' + 'n_gen_' +str(n_gen) + '_seed_'+ str(seed)
    
    
