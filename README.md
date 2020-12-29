# scikit-learn-linear-regression

In this project we will implement linear regression using scikit-learn library. We will see things can be done in a few line.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" alt="sklearn" width="400" height="200">

Scikit-learn is a free software machine learning library for the Python programming language.It features various classification, regression and clustering algorithms including support vector machines (SVM), random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

We will use sklearn diabetes dataset that's aleardy existing in sklearn library.
Our data will be splitted into two types of data. The first one is the training data that we will use to train our model, the second is the test data that will help us evaluate our model and see its accuracy. 

# Libraries

    from sklearn import datasets                                  # For using existing sklearn datasets 
    from sklearn.linear_model import LinearRegression             # desired model : linear regression
    from sklearn.model_selection import train_test_split          # split data to training ones and test ones
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score      # model evaluation
    
# Data
    # to load a dara name 'ABC' for example : datasets.load_ABC() and it's a numpy array
    my_data = datasets.load_diabetes()                                         
    
    
    # let's convert it into a pandas dataframe to make it human readable
    dataframe = pd.DataFrame(data= np.c_[my_data['data'], my_data['target']],  
                         columns= my_data['feature_names'] + ['target'])
                         
This dataset contains physiological data collected on 442 patients and as a corresponding
target an indicator of the disease progression after a year. The physiological data occupy
the first 10 columns with values that indicate respectively the following:

• Age <br/>
• Sex <br/>
• Body mass index <br/>
• Blood pressure <br/>
• S1, S2, S3, S4, S5, and S6 (six blood serum measurements) <br/>

# Display dataset:

<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--_wjflnM3--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://snipboard.io/L1SrbR.jpg">

As for the indicators of the progress of the disease, that is, the values that must correspond to the results of our predictions, these are obtainable by means of the target attribute as shown in the following:

<table>
  <tr>
    <th>Target</th>
  </tr>
  <tr>
    <td>151.0</td>
  </tr>
  <tr>
    <td>75.0</td>
  </tr>
  <tr>
    <td>141</td>
  </tr>
  <tr>
    <td>206.0</td>
  </tr>
  <tr>
    <td>...</td>
  </tr>
  
  <tr>
    <td>220.0</td>
  </tr>
  <tr>
    <td>57.0</td>
  </tr>
</table>

   


# variables :
    x = my_data.data
    y = my_data.target
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.01, random_state=0)  #split the data       

# Training and predicting
    
    regression_model = LinearRegression()             # create object of our model

    
    regression_model.fit(X_train,y_train)             # Training model

    
    y_pred = regression_model.predict(X_test)         # Prediction


# Model evaluation
    R_train=regression_model.score(X_train,y_train)*100
    R_test=regression_model.score(X_test,y_test)*100
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.

<img src="https://miro.medium.com/max/966/1*lqDsPkfXPGen32Uem1PTNg.png" width="400" height="200">    


R-squared,R² or r², is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

<img src="https://ashutoshtripathicom.files.wordpress.com/2019/01/rsquarecanva2.png" width="400" height="250">

# theta parameters
    intercept = regression_model.intercept_
    coefs = regression_model.coef_
   
θ0 is the intercept   <br/>
[θ1,..,θn] is coefs, with n is number of features


# Results :

R_test, The accuracy of this model is: 64.60 % <br/>
R mean squared error: 2647.3835603036587<br/>
R squared: 0.6460262867555651
