#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('playerprices.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#Pre processing to convert all relevant data into numberical form
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# THIS only to used to debug or see values in variable  explorer
df = pd.DataFrame(X)

#split into 2 parts
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0,test_size=0.25)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

#append all the prediction and actual price in tuple for visualizing in bar chart
predicted=[]
real=[]
#change this variable to take more or less values for visulazing
batchsize=20
for i in range(0,batchsize):
    predicted.append(int(y_pred[i]))
    real.append(int(y_test[i]))


predicted=tuple(predicted)
real = tuple(real)
# data to plot
n_groups = batchsize

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, predicted, bar_width,
alpha=opacity,
color='b',
label='Predicted Price')

rects2 = plt.bar(index + bar_width, real, bar_width,
alpha=opacity,
color='g',
label='Original Price')

plt.xlabel('Players Index(Ignore the labels)')
plt.ylabel('Prices in Rupees')
plt.title('Prediction Prices vs Actual Price')
plt.legend()

plt.tight_layout()
plt.show()