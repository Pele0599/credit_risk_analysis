import numpy as np 
from sklearn.linear_model import LinearRegression

class linear_regression:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.reg = None
        return 
    def fit(self):
        reg = LinearRegression().fit(self.X, self.y)
        self.reg = reg
    


