from pickletools import optimize
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import OrderedDict
from torch import seed
from torch.utils.data import DataLoader
from read_data import *
from models.linear_regression import linear_regression 
from sklearn.model_selection import train_test_split
from models.neural_networks.data_loader import Data
from models.neural_networks.neural_network import *
from models.neural_networks.loss_fn import *
from sklearn.metrics import f1_score
# Initialize the data 
datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
X,y = get_data(datapath)
x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=0.1)
data=Data(x_train, y_train, x_test, y_test)

loader=DataLoader(dataset=data,batch_size=32)




# Train the Neural netowrk 
# and get the results 
neural_net = NeuralNet(data.X_train.shape[1], 60, 2).to('cpu')
loss_fn = cross_entropy_loss_2d()
EPOCHS = 1500
LR = 0.0001
optimizer = torch.optim.Adam(neural_net.parameters(), lr=LR)
cont_loss = train(neural_net, loader, loss_fn, optimizer, EPOCHS)
plt.plot(cont_loss)
plt.show()
y_pred = neural_net(data.X_test.float()).detach().cpu().numpy()
print(y_pred)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0

y_test_np = data.Y_test.detach().cpu().numpy()
f1 = f1_score(y_test_np, y_pred)
print(f1)
plt.plot(y_pred)
plt.show()
# STORE THE MODEL 


# Results for linear regression 
linear_regressor = linear_regression(x_train,y_train)
linear_regressor.fit()


# Results for XGBOoost

# Results for symbolic regression 












