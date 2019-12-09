import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# creating a list with the header since it is not included in the file.
header = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
df = pd.read_csv("crx.data", header = None, names = header) # putting data into a dataframe
df.dropna(inplace=True) # dropping the missing values

# I will create a list of the categorical variables for future use when creating the dummy variables
# specifically to drop from the new data frame after adding in the dummy variables
# Also by removing A16_-, A16_+ is coded as a 1 for a + and 0 for a - and that is what ill be using for prediction
categorical_vars=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13","A16", "A16_-"]
# now I get dummy variables from categorical_vars, dropping A16_- as it is not applicable here as mentioned above

one_hot_encoding = pd.get_dummies(df[categorical_vars[:10]])
df = pd.concat([df, one_hot_encoding], axis = 'columns') # adding dummy variables to data frame
df = df.drop(categorical_vars, axis=1) # dropping the original categorical variables as well as the A16_-
# I will create a temporary data frame and drop the dependent variable to create my X variables
temp=df.drop("A16_+", axis=1)
Y = df["A16_+"] # setting the dependent variable
inputs=df[list(temp.columns.values)] # setting the X values
# creating the training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, Y, test_size=0.3, random_state=1)

# Tree with criterion gini and max_depth none
tree = DecisionTreeClassifier(criterion="gini") # creating the decision tree object
tree.fit(X_train, y_train) # fitting the tree with the training data
y_pred = tree.predict(X_test) # using the fit to predict the X test data
print("Accuracy with criterion gini and no max depth:",metrics.accuracy_score(y_test, y_pred)) # printing the accuracy score

# Tree with criterion entropy and max_depth none
tree2 = DecisionTreeClassifier(criterion="entropy") # creating the decision tree object
tree2.fit(X_train, y_train) # fitting the tree with the training data
y_pred2 = tree2.predict(X_test) # using the fit to predict the X test data
print("Accuracy with criterion entropy and no max depth :",metrics.accuracy_score(y_test, y_pred2)) # printing the accuracy score

# creating depth and criteria lists to loop through and fit the various models
depth = [ 2, 3, 4]
criteria = ["gini", "entropy"]

for d in depth:
    for c in criteria:
        tree = DecisionTreeClassifier(criterion=c, max_depth = d)  # creating the decision tree object
        tree.fit(X_train, y_train)  # fitting the tree with the training data
        y_pred = tree.predict(X_test)  # using the fit to predict the X test data
        print("Accuracy with criterion",c, "and max _depth", d, ":", metrics.accuracy_score(y_test, y_pred))  # printing the accuracy score
