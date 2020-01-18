# TradingBinary Polynomial Regression
___
Algorithm to predict the best parameters in a polynomial regression and verify if it is a good method to trade binary options

## Assumption:
The assumption is that the market has trendings, and since a polynomial regression has a good fit in the ends of the interval, the next predicted point is supposed to follow the trend.

## Challenge:
To predict if the next closing price will be higher or lower, based on a polynomial regression model.

## Method
Using the polynomial regression model, assume that the next closing price will be the value predicted by the model.

If the predicted price is lower, the guess is that the price will go down.

If the predicted price is higher, the guess is that the price will go up.

_The following parameters will be under control:

- The size of the window in wich the regression line will be calculated.
- The degree of the polynomial regression

## Results
Running the algorithm with a window size varying from 5 to 50 and degree varying from 2 to 7, the best result was:

Win ratio: 52.09%\
Window size: 14\
Degree: 4

___
## Prediction Plot
Here we see all the closing prices as the small blue dots. The red ones are the predictions without profit, and the green stars as the profit ones.
![alt tag](/img_plot/polynomial_predictions.png)

## Conclusion
Despite the idea that a polynomial regression would predict the price based on the trend, it was possible to realize that, even with lots entry points, and lots of profits, the overnight variation does not necessarily follow the general trend, wich is observed over a longer period. So for this same dataset a simple linear regression proved more efficient, but with much more less entry points although.
