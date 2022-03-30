import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep = ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

# The label we want to make predictions for
predict = "G3"

# array that defines all the lables we are using to make our predictions which would be our input
x = np.array(data.drop([predict], axis = 1))
# array of the labels we are predicitng for
y = np.array(data[predict])


# for a CNN it was images as input then label as output now its all the other labels as the input and the G3 label as the output
# This splits the our input/output arrays to training arrays and testing arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# creates a linear regression model
linear = linear_model.LinearRegression()

# finds the best fit line for the data
linear.fit(x_train, y_train)

# Saves the model for us so we dont have to keep training it all the time
with open("studentModel.pickle", "wb") as f:
    pickle.dump(linear, f)

# Loads the model
pickle_in = open("studentModel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)

print(acc)
print(linear.coef_)
print(linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.show()

