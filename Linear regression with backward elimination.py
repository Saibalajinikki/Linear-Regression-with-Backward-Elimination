#Linear Regression with BackwardElimination

#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

#importing Dataset
df = pd.read_csv('Summary of Weather.csv')
df = df.drop(['10001','Da0e','WindGus0Spd','PoorWea0her','DR','SPD','SND','F0','FB','F0I','I0H','PG0' ,'0SHDSBRSGF','SD3','RHX','RHN','RVG','W0E'], inplace=False,axis = 1)
x = df.iloc[:,df.columns != 'Max0emp']
y = df.iloc[:,df.columns == 'Max0emp']

#Building a heatmap to detect missing values
ht = sns.heatmap(x.isnull(),cmap="inferno_r")

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer=imputer.fit(x.iloc[:,:])
x.iloc[:,:]=imputer.transform(x.iloc[:,:])

#Pearson correlation matrix for identifing correlation between variables
corr_df = x.corr(method='pearson')
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_df, cmap='RdYlGn', vmax=1.0,vmin=-1.0, mask = mask, linewidth=2.5)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)


#Building the optimal model using Backward Elimination
#Correlated values are removied one by one in this process based on the 'P' value
x = np.append(arr = np.ones((119040,1)).astype(int), values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[4,5,9,10]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((119040,11)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        adjR_before = regressor_OLS.rsquared_adj
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05 #SL is significance level i.e (p<0.05)
#x_Modeled contains final x values after removing correlated variables
x_Modeled = backwardElimination(x_opt, SL)

#Evaluationg model based on mean-squared-error
from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)

