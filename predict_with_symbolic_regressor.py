from models.symbolic_regression.fastr_symbolic_regressor import *
from read_data import *
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt 
from models.symbolic_regression.gplearn import *

# Load fast symbolic regression model 
# sr = SymbolicRegression()
# POP_SIZE=300
# N_GENERATIONS = 1000
# SEED = 42
# model_name = get_pkl_file_name_fastsr(POP_SIZE,N_GENERATIONS,seed=SEED)
# sr.load(model_name)

# sr.print_best_individuals()
datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
X,y = get_data_single_outpu(datapath)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)
# y_pred = sr.predict(X_test.to_numpy())
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred < 0.5] = 0
# print(y_pred)
# plt.scatter(np.arange(len(y_pred)),y_pred)
# plt.scatter(np.arange(len(y_test)) + 0.01,y_test)
# plt.show()

# Load pre trained gplearn model 
POP_SIZE=1000
N_GENERATIONS = 5000
SEED = 42
model_path = '/Users/paolovincenzofreieslebendeblasio/finpack/models/symbolic_regression/trained_symbolic_models/gplearn/loan_type/gplearn_pop_sz_15_n_gen_50_seed_42_pars_coeff_0.01.pkl'

# gplearn_reg = load_gplearn_model(POP_SIZE, 
#     N_GENERATIONS, 
#     SEED)

y_pred = gplearn_reg.predict(X_test.to_numpy())

print(gplearn_reg._program)

plt.plot(y_pred)
plt.show()