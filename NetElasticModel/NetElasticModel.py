import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# creating one data frame for the red wine csv and one for the white wine csv
df_red = pd.read_csv("winequality-red.csv", sep=";")
df_white = pd.read_csv("winequality-white.csv", sep=";")

# creating a new variable that will be white or red for the wine
df_red["color"] = 1
df_white["color"] = 0

# combining the two dataframes by appending the white wine df to the red wine df
df = df_red.append(df_white)

# setting the X and Y columns
Y = df["quality"]
X = df.drop(["quality"], axis = 1)

# creating class instance
net = ElasticNet(alpha=0.1, l1_ratio = 0.5)

# finding mse using cross validation
score = cross_val_score(net, X, Y, cv=10, scoring="neg_mean_squared_error")
print(score)
average_mse = score.mean()
print(average_mse)

"""
Average MSE = 0.5997
I decided to use an Elastic net model here for a couple of reasons. First the winequality.names files states that some of the attributes 
may be correlated so we cannot use a linear regression model as collinearity violates an assumption of the model. Also the winequality.names
file states that they are not sure if all the input variables were relevant, because of that we need to use either lasso or an elastic net 
model as the ridge regression model cannot shrink coefficients all the way to zero and remove them from the model. Between lasso and and 
elastic net I decided to use an elastic net model to combine ridge and lasso to get the benefits from both. Overall, the MSE value is very low 
meaning the model fits the data well. 
"""






