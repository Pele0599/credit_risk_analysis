import pandas as pd 
import os, sys
sys.path.append(os.path.join(sys.path[0], '..'))
from data.datapaths import datapaths
from sklearn.decomposition import PCA
from read_data import * 
import matplotlib.pyplot as plt 

datapath = datapaths['credict_card_acceptance']
X,y = get_credit_card_data_single_output(datapath, 
    normalize_data=True)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
    ,columns = ['principal component 1', 'principal component 2', 'principal component 3'])


y_df = pd.DataFrame(data = y.to_numpy(), columns=['accepted'])
finalDf = pd.concat([principalDf, y_df['accepted']], axis = 1)

fig = plt.figure(figsize = (8,8))
print(X)
ax =plt.axes(projection = "3d")
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['accepted'] == target
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
