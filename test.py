from cProfile import label
from pickletools import optimize
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import OrderedDict
from torch import seed
from torch.utils.data import DataLoader
from read_data import *
from models.polynomial_regression.linear_regression import linear_regression 
from sklearn.model_selection import train_test_split
from models.neural_networks.data_loader import Data
from models.neural_networks.neural_network import *
from models.neural_networks.loss_fn import *
from sklearn.metrics import f1_score
# Initialize the data 
datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
X,y = get_data(datapath)
x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.2)
data=Data(x_train, y_train, x_test, y_test)

loader=DataLoader(dataset=data,batch_size=128)




# Train the Neural netowrk 
# and get the results 
# neural_net = NeuralNet(data.X_train.shape[1], 50, 2).to('cpu')
# loss_fn = cross_entropy_loss_2d()
# EPOCHS = 500
# LR = 0.0001
# optimizer = torch.optim.Adam(neural_net.parameters(), lr=LR)
# cont_loss = train(neural_net, loader, loss_fn, optimizer, EPOCHS)
# plt.plot(cont_loss)
# plt.show()
# y_pred = neural_net(data.X_test.float()).detach().cpu().numpy()
# plt.scatter(np.arange(len(y_pred[:,1])), y_pred[:,1], label='Pred')
# plt.scatter(np.arange(len(y_pred[:,1])), y_test[:,1], label = 'Truth')

# plt.legend(loc='upper right')
# plt.show()

# y_pred[:,1][ y_pred[:,1] > 0.5] = 1
# y_pred[:,1][ y_pred[:,1] < 0.5] = 0

# y_pred[:,0][ y_pred[:,0] < 0.5] = 0
# y_pred[:,0][ y_pred[:,0] > 0.5] = 1

# y_test_np = data.Y_test.detach().cpu().numpy()
# f11 = f1_score(y_test_np[:,1], y_pred[:,1])
# f10 = f1_score(y_test_np[:,0], y_pred[:,0])
# print((f11 + f10) / 2, 'f1 score')

# STORE THE MODEL 


# Results for linear regression 
linear_regressor = linear_regression(x_train,y_train)
linear_regressor.fit()
preds = linear_regressor.reg.predict(x_test)
preds[:,0][preds[:,0] > 0.5] = 1
preds[:,0][preds[:,0] < 0.5] = 0
plt.plot(preds[:,0][abs(preds[:,0]) < 2])
plt.show()
# Results for XGBOoost

# Results for symbolic regression 












