import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self):
        self.weights= np.array([0])

    def fit(self,X,y,lr=0.01,epoches=1000):
        arr= np.array([1]*len(X))
        X= np.column_stack((arr,X))
        y= y.reshape((-1,1))
        n= len(X)
        m= len(X[0])
        
        self.weights= np.zeros((m,1))
        
        for epoch in range(epoches):
            predicted= np.zeros(n)
            
            predicted= X@self.weights
            loss= ((predicted-y)**2).sum().item()/n
            
            grad= 2/n*(X.T@(predicted-y))
            self.weights-=(lr*grad)
                
            if epoch==0:print(f'Epoch {epoch+1} MSE: {loss}')
            elif epoch%50==0:print(f'Epoch {epoch} MSE: {loss}')
            
    def predict(self,X):
        arr= np.ones((len(X),1))
        X= np.column_stack((arr,X))
        prediction= X@self.weights
        return prediction.flatten()
            
class StandarScaler():
    def __init__(self):
        self.mu=0
        self.sd=1
        
    def fit_transform(self,X):
        self.mu= X.mean(axis=0)
        self.sd= X.std(axis=0)
        return (X-self.mu)/(self.sd+ 1e-8)
    
    def transform(self,X):
        return (X-self.mu)/(self.sd+ 1e-8)
    