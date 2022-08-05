import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
from data.datapaths import datapaths
from models.symbolic_regression.fastr_symbolic_regressor import *
from models.symbolic_regression.gplearn import *
from sklearn.model_selection import train_test_split
from read_data import *
import matplotlib.pyplot as plt 


datapath = datapaths['credit_card_acceptance']
X,y = get_credit_card_data_single_output(datapath)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)

# Parameters for the symbolic regression models
POP_SIZE=1000
N_GENERATIONS = 500
SEED = 42
INIT_DEPTH=(1, 8)
MIN_DEPTH_INIT=2
MAX_DEPTH_INIT=8
PARSIMONY_COEFF = 0.00001 # By having a small parsimony coefficient, we increase the probability
# that the final symbolic expression is longer 
SAVE_FILE = get_pkl_file_name_fastsr(N_GENERATIONS,POP_SIZE,SEED)

# Fast symbolic regression 
# sr = fastsr(ngen=N_GENERATIONS, pop_size=POP_SIZE, 
#     min_depth_init=MIN_DEPTH_INIT,  max_dept_init = MAX_DEPTH_INIT, 
#     crossover_probability=0.2,seed=SEED)
# GPLearn 
gplearn_reg = gplearn_regressor(ngen = N_GENERATIONS, pop_size=POP_SIZE, 
    parsimony_coefficient=PARSIMONY_COEFF, init_depth=INIT_DEPTH)

gplearn_reg.fit(X_train.to_numpy(), y_train)
save_gplearn_model_pkl(POP_SIZE,
    N_GENERATIONS, 
    SEED,
    gplearn_reg.est,
    dataset_type='loan_type')

# Save trained gplearn model locally 
# sr.save(SAVE_FILE)

score = gplearn_reg.score(X_test.to_numpy(), y_test)

print('Score: {}'.format(score))

results = gplearn_reg.predict(X_test.to_numpy())

plt.scatter(np.arange(len(results)) + 0.1,results)
plt.scatter(np.arange(len(results)),y_test,marker='^')
plt.show()


