from sklearn import datasets                         # For using existing sklearn datasets 
from sklearn.linear_model import LinearRegression    # desired model
from sklearn.model_selection import train_test_split # split data to training ones and test ones 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score # model evaluation

my_data = datasets.load_diabetes() 
dataframe = pd.DataFrame(data= np.c_[my_data['data'], my_data['target']],
                         columns= my_data['feature_names'] + ['target']) # store data as pandas dataframe
x = my_data.data
y = my_data.target

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.01, random_state=0)   # splitting data

regression_model = LinearRegression()  # create object of our model

# Training model
regression_model.fit(X_train,y_train)

# Prediction
y_pred = regression_model.predict(X_test)

# Model evaluation
R_train=regression_model.score(X_train,y_train)*100
R_test=regression_model.score(X_test,y_test)*100
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# theta parameters
intercept = regression_model.intercept_
coefs = regression_model.coef_


print(dataframe)
print('The accuracy of this model is:', "{:.2f}".format(R_test),'%')
print('R mean squared error:',rmse)
print('R squared:',r2)