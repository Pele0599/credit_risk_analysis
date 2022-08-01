import pandas as pd 
import numpy as np 

def get_data_neural_network(datapath,normalize_data=True):

    data = pd.read_csv(datapath)
    if normalize_data:
        y = data.pop('Bankrupt?')
        y2d = np.zeros((len(y),2))
        # turn a 0,1 into [0,1] or [1,0] depending 
        for i, val in enumerate(y):
            if val == 0:
                y2d[i] = [0,1]
            else:
                y2d[i] = [1,0]
        
        normalized_df=(data-data.mean())/data.std()
        return normalized_df, y2d
    return data

def get_data_single_outpu(datapath, normalize_data = True):
    data = pd.read_csv(datapath)
    if normalize_data:
        y = data.pop('Bankrupt?')
        normalized_df=(data-data.mean())/data.std()
        return normalized_df, y
    return data

def get_data_with_synthetic(datapath, n_synthetic_data):
    data = pd.read_csv(datapath)
    bankruptcy_rows = data[data['Bankrupt?'] == 1]
    print(bankruptcy_rows.mean())
    print(bankruptcy_rows.std())
    for i in range(n_synthetic_data):
        data = pd.concat([data, bankruptcy_rows], axis=0)
    return data 



# datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
# print(add_synthetic_data(datapath,5))


