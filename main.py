import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Read the dataset, set the index as the Date column,
#converts the index’s type from “object” to “datetime64[ns]”
df = pd.read_csv('Google.csv', usecols=['Date', 'Close'], index_col=[0], 
                 date_parser=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))


fit_params ={}
def best_fit(w, degree):
    def pol_reg(y):
        """Receive a sequence of values, ndarray, and apply a polynomial regression,
        based on a x axis of equaly espaced numbers.
        Return: the predicted value for the next number.
        """
        #   set the x axis based on the lenght of the ndarray
        x = np.array(list(range(len(y)))).reshape(-1,1)
        
        #   fit a polynomial regression
        poly_reg = PolynomialFeatures(degree=degree)
        X_poly = poly_reg.fit_transform(x)
        regressor = LinearRegression()
        regressor.fit(X_poly, y)
        y_pred = regressor.predict(poly_reg.fit_transform([[w]]))
    
        return y_pred
    
    #create the column with the predicted values for each date
    df['Predicted'] = df.Close.rolling(w).apply(pol_reg, 'raw=True').shift(1)
    
    #create a column with the diference of price expected for the next day
    df['Expected_diff']= df.Predicted - df.Close.shift()
    
    #create a column with the actual price difference
    df['Actual_diff']= df.Close.diff()
    
    #create a bool column to verify if the expected variation was positive
    df['Expected_pos'] = df.Expected_diff > 0
    
    #create a bool column to verify if the actual variation was positive
    df['Actual_pos'] = df.Actual_diff > 0
    
    #create a Profit column verifying if the sign of the prection was realized
    df['Profit'] = df.Expected_pos == df.Actual_pos
    
    #count the true profits on the rows where the predicted values are above zero
    win = sum(df[df.Predicted > 0].Profit)
    
    #count the total number of valid rows
    total = len(df[df.Predicted > 0])
    
    #calculate the ration between the wins and the total
    ratio = (win/total)

    return ratio


def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]
 
########################################################################
#Define parameters and run the model

# Variable that define the max size of the window to run the model
max_window = 50

# Variable that define the max polynomial degree
max_degree = 7

#loop to create a dictionary with the results for each parameter
# window defined between 5 and <max_window
for w in range(5,max_window):
#    degree defined between 2 and <max_degree
    for d in np.arange(2, max_degree):
        ratio = best_fit(w, d)
        fit_params[(w, d)] = ratio

#get the dict key of best fit. It is a tuple (window, degree)
best_fit = keywithmaxval(fit_params)
##############################################################

print('The best won ratio was: ' + str(round(fit_params[best_fit]*100, 2)) +'%')
print('Window:', best_fit[0])
print('degree:', best_fit[1])




###########To plot the predictions
#best_predicted = best_fit(best_fit[0], best_fit[1])
#plt.scatter(df.index, df.Close, s=0.5, label="Close")
#plt.scatter(df.index[df.Predicted > 0], df.Predicted[df.Predicted > 0], s=1, color='red', marker= "v", label="Predicted")
#plt.scatter(df.index[(df.Profit==True) & (df.Predicted > 0)], df.Predicted[(df.Profit==True) & (df.Predicted > 0)], s=3, 
#                     marker= "*", color='green', label="Profit")
#plt.legend()
#plt.xlabel("Date")
#plt.ylabel("Closing Price")
#plt.show()


#Test plot to see the behaviour of the curves
################
#w=14
#dg=4
#x = np.array(list(range(w))).reshape(-1,1)
#regressor = LinearRegression()
#regressor.fit(x, df.iloc[0:w].Close)
#y_pred_linear = regressor.predict(x)
#y_lin = regressor.predict([[(w)]])
#
#poly_reg = PolynomialFeatures(degree=dg)
#X_poly = poly_reg.fit_transform(x)
#
#regressor2 = LinearRegression()
#regressor2.fit(X_poly, df.iloc[0:w].Close)
#y_pred_poly = regressor2.predict(poly_reg.fit_transform(x))
#y_pol =  regressor2.predict(poly_reg.fit_transform([[(w)]]))
#
#plt.plot(x, df.iloc[0:w].Close, color='red')
#plt.plot(x, y_pred_linear, color='blue')
#plt.plot(x, y_pred_poly, color='green')
#plt.scatter(w, df.iloc[w].Close, color='red')
#plt.scatter(w, y_pol, color='green')
#plt.scatter(w, y_lin, color='blue')

    
