# Linear Regression with Torch

Toy example of linear regression model made with Torch. Three main components of the model are:
 - **nn.Linear** - linear layer of neural network that performs (mx + b) operation
 - **nn.MSELoss** - Mean Squared Loss function
 - **torch.optim.SGD** - Stochastic Gradient Descend


```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import pandas as pd
```

## 1. Model


```python
class TorchLinearRegression:
    
    def __init__(self, lr=0.001):
        # learning rate
        self.lr = lr
                
    def fit(self, X, y, epochs=60):
        # array to tensor
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32)
        # model parameters
        input_size = X.shape[1]
        output_size = y.shape[1]
        # model, loss, optimizer
        self.model = nn.Linear(input_size, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr) 
        # training
        for i in range(epochs):
            # forward
            outputs = self.model(X)
            # compute loss
            loss = criterion(outputs, y)
            # set gradients to 0
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
            
    def predict(self, X):
        X = torch.from_numpy(X).type(torch.float32)
        y_pred = self.model(X).detach().numpy()
        return y_pred
```

## 2. Data

Features scaled to [0, 1]


```python
df = pd.DataFrame(load_boston()['data'], columns=load_boston()['feature_names'])
df['target'] = load_boston()['target']
data = MinMaxScaler().fit_transform(df)

X = data[:, :-1] # all the feautres
y = data[:, -1:] # target

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Training


```python
model = TorchLinearRegression(lr=0.1)
model.fit(X, y, 500)
y_pred = model.predict(X)
```

## 4. Evaluation


```python
mean_squared_error(y, y_pred)
```




    0.011920771594821645




```python
m = nn.Linear(1, 1)
```
