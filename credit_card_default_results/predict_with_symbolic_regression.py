import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
from models.symbolic_regression.fastr_symbolic_regressor import *
from read_data import *
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt 
from models.symbolic_regression.gplearn import *
from models.model_paths import model_paths
from data.datapaths import datapaths
from sklearn.metrics import f1_score, roc_auc_score
import graphviz

model = 'gplearn'
dataset = 'credit_card_acceptance'
print(model_paths, 'This is model paths')
model_path = model_paths[dataset][model]
data_path = datapaths['credit_card_acceptance']

X,y = get_credit_card_data_single_output(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)

# Load pre trained gplearn model 
POP_SIZE=1000
N_GENERATIONS = 500
SEED = 42
model_path = '/Users/paolovincenzofreieslebendeblasio/finpack/models/symbolic_regression/trained_symbolic_models/gplearn/loan_type/seed_42/gplearn_pop_sz_45_n_gen_30_pars_coeff_0.1.pkl'
# gplearn_reg = load_gplearn_model(POP_SIZE,
#     N_GENERATIONS, 
#     SEED,
#     model_path)
with open(model_path, 'rb') as f:
    gplearn_reg = pickle.load(f)

print(gplearn_reg.get_params(deep=True), 'MODEL')


# y_pred = gplearn_reg.predict(X_test.to_numpy())

# print(gplearn_reg._program)

# plt.scatter(np.arange(len(y_pred)),y_pred,c='r')
# plt.scatter(np.arange(len(y_pred)) + 0.2,y_test,c='g')
# plt.show()

# print('The f1 score is:', f1_score(y_test, y_pred))
# print('AUC score is', roc_auc_score(y_test, y_pred))

# from IPython.display import Image
# import pydotplus
# from graphviz import Source

# # View the output as a graph 
# dot_data = gplearn_reg._program.export_graphviz()
# graph = graphviz.Source(dot_data)
# graph.view()
