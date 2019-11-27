import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
from matplotlib import style
import pickle

# type = pandas.core.frame.DataFrame
data = pd.read_csv('student-mat.csv', sep=";")

# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

label = "G3"

arr_x = np.array(data.drop([label], 1))
arr_y = np.array(data[label])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(arr_x, arr_y, test_size=0.1)

# Train till the maximum accuracy
# best = 0
#
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(arr_x, arr_y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#
#     print('Accuracy: ', accuracy)
#
#     if accuracy > best:
#         best = accuracy
#         print('Best: ', best)
#         with open('student-model.pickle', 'wb') as f:
#             pickle.dump(linear, f)

# Open best trained model
pickle_in = open('student-model.pickle', 'rb')
linear = pickle.load(pickle_in)

print('Coefficient:\n', linear.coef_)
print('Intercept:\n', linear.intercept_)

predictions = linear.predict(x_test)

print('======= PREDICTIONS ==========')
print('G3 (Predicted)', '   [TRAINING DATA]', 'G3 (Actual)')
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'failures'
style.use('ggplot')
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
