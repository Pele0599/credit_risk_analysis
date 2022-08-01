from models.symbolic_regression.fastr_symbolic_regressor import *
from sklearn.model_selection import  KFold, train_test_split
from read_data import *
import matplotlib.pyplot as plt 
datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
X,y = get_data_single_outpu(datapath)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)

# Parameters for the symbolic regression models
POP_SIZE=500
N_GENERATIONS = 3000
SEED = 42
MIN_DEPTH_INIT=2
MAX_DEPTH_INIT=6
SAVE_FILE = get_pkl_file_name_fastsr(N_GENERATIONS,SEED)

sr = SymbolicRegression(ngen=N_GENERATIONS, pop_size=POP_SIZE, 
    min_depth_init=MIN_DEPTH_INIT,  max_dept_init = MAX_DEPTH_INIT, 
    crossover_probability=0.2,seed=SEED)

sr.fit(X_train.to_numpy(), y_train)
sr.save(SAVE_FILE)

score = sr.score(X_test.to_numpy(), y_test)

sr.save()

print('Score: {}'.format(score))

results = sr.predict(X_test.to_numpy())

plt.scatter(np.arange(len(results)),results)
plt.scatter(np.arange(len(results)),y_test,marker='^')
plt.show()

def save(self, filename)
