import pandas as pd 
from sklearn.decomposition import PCA
from read_data import * 
import matplotlib.pyplot as plt 

datapath = '/Users/paolovincenzofreieslebendeblasio/finpack/data/data_company_bankruptcies.csv'
X,y = get_data_single_outpu(datapath, normalize_data=True)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
    ,columns = ['principal component 1', 'principal component 2', 'principal component 3'])


y_df = pd.DataFrame(data = y.to_numpy(), columns=['bankrupt'])
print(y_df)
finalDf = pd.concat([principalDf, y_df['bankrupt']], axis = 1)
print(finalDf)
fig = plt.figure(figsize = (8,8))

ax =plt.axes(projection = "3d")
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['bankrupt'] == target
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1']
               ,finalDf.loc[indicesToKeep, 'principal component 2']
               ,finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

print('Explained variance ', pca.explained_variance_ratio_)
print(pca.components_, 'PCA components')
