import pandas as pd 
import numpy as np 
import category_encoders as ce
def get_data_neural_network(datapath,normalize_data=True):
    '''
        Gets bankruptcy data for different 
    '''
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

def get_credit_card_data_single_output(datapath, normalize_data = True):
    data = pd.read_csv(datapath)
    if normalize_data:
        y = data.pop('credit.policy')
        purpose = data.pop('purpose')
        ce_OHE = ce.OneHotEncoder(cols=['purpose'])
        purpose_OHE = ce_OHE.fit_transform(purpose) 
        # Transform categorical variables
        # Using one hot encoding 
        normalized_df=(data-data.mean())/data.std()
        X_df = pd.concat([normalized_df,purpose_OHE],axis = 1)
        return X_df, y
    return data


# datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
# print(add_synthetic_data(datapath,5))


