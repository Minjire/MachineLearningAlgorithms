import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("car.data")
print(data.head())

# object to encode labels in data into appropriate integer values
le = preprocessing.LabelEncoder()
# method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# zip() method convert list values to tuple
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# create KNN classifier and specify how many neighbours to look for
model = KNeighborsClassifier(n_neighbors=9)
# train model
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predict = model.predict(x_test)
names = ["unacc", 'acc', 'good', 'vgood']
for i in range(len(predict)):
    print("Predicted: ", names[predict[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
    # see neighbours of a given point
    n = model.kneighbors([x_test[i]], 9, True)
    print("N: ", n)


