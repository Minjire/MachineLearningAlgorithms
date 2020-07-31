# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# %%
mydata = pd.read_csv('drug200.csv')
mydata.head()

# %% Preprocessing
# select features
X = mydata[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[:5])

# convert categorical values to numerical
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', "NORMAL", 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[:5])

# fill target variable
y = mydata["Drug"]
print(y[:5])

# %% Setting up Decision Tree
from sklearn.model_selection import train_test_split

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print(f'{X_train.shape} {y_train.shape}')
print(f'{X_test.shape} {y_test.shape}')

# modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)

drugTree.fit(X_train, y_train)
predTree = drugTree.predict(X_test)

print(predTree[:5])
print(y_test[:5])

# %% Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt

print(f'DecisionTree\'s Accuracy: {metrics.accuracy_score(y_test, predTree)}')

# %% Visualize the Tree
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

dot_data = StringIO()
filename = 'drugtree.png'
featureNames = mydata.columns[:5]
targetNames = mydata['Drug'].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(y_train),
                           filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')

# %%
y = pydotplus.find_graphviz()
print(y)
