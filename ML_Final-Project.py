# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

# %% Load Data
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset'
                 '/ML0101EN_EDX_skill_up/cbb.csv')
print(df.head())

# add a column that will contain "true" if the wins above bubble are over 7 and "false" if not
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

# %% Data Visualization and Preprocessing
# filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, and the Final Four in the post season
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
print(df1.head())

print(df1['POSTSEASON'].value_counts())

# %% PLot columns
import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col='windex', hue='POSTSEASON', palette='Set1', col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec='k')

g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

# %% Feature selection/Extraction
# plot adjusted defense efficiency
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col='windex', hue='POSTSEASON', palette='Set1', col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec='k')
g.axes[-1].legend()
plt.show()

# We see that this data point doesn't impact the ability of a team to get into the Final Four.

# %% Convert Categorical variable to numerical
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
# 13% of teams with 6 or less wins above bubble make it into the final four while 17% of teams with 7 or more do.

# convert wins above bubble (winindex) under 7 to 0 and over 7 to 1
df1['windex'].replace(to_replace=['False', 'True'], value=[0, 1], inplace=True)
print(df1.head())

# %% Feature Selection
# features
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
         'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
         '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
print(X[:5])
# labels
y = df1['POSTSEASON'].values
print(y[:5])

# %% Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[:5])

# split data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print(f"Train set: {X_train.shape} {y_train.shape}")
print(f"Validation set: {X_val.shape} {y_val.shape}")

# %% Classification
# KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

k = 5
knn_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(knn_model)

knn_pred = knn_model.predict(X_val)
print(knn_pred[:5])

print(f"Train set accuracy: {round(accuracy_score(y_train, knn_model.predict(X_train)), 4)}")
print(f"Validation set accuracy: {round(accuracy_score(y_val, knn_pred), 4)}")

Ks = 15
metrics_knn = np.zeros(Ks)
for i in range(Ks):
    Ks_model = KNeighborsClassifier(n_neighbors=i + 1).fit(X_train, y_train)
    Ks_pred = Ks_model.predict(X_val)
    metrics_knn[i] = round(accuracy_score(y_val, Ks_pred), 4)

metrics_knn

# %% Decision Tree
from sklearn.tree import DecisionTreeClassifier

trees_depth = 10
best_metrics_trees = 0

for i in range(trees_depth):
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=i + 1).fit(X_train, y_train)
    tree_pred = tree_model.predict(X_val)
    tree_metric = round(accuracy_score(y_val, tree_pred), 4)
    if tree_metric > best_metrics_trees:
        best_metrics_trees = tree_metric
        best_depth = i + 1

print(f"Best results: {best_metrics_trees} with minimum depth: {best_depth}")

# %% SVM
from sklearn import svm
from sklearn.metrics import f1_score, jaccard_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# best_f1_score = 0
# best_jaccard_score = 0
best_accuracy_svm = 0

for k in kernels:
    svm_model = svm.SVC(kernel=k).fit(X_train, y_train)
    svm_pred = svm_model.predict(X_val)
    svm_acc = accuracy_score(y_val, svm_pred)
    # svm_f1 = round(f1_score(y_val, svm_pred, average='weighted'), 4)
    # svm_jac = round(jaccard_score(y_val, svm_pred, average='weighted'), 4)
    # if svm_f1 > best_f1_score:
    #     best_f1_score = svm_f1
    #     best_kernel_f1 = k
    # if svm_jac > best_jaccard_score:
    #     best_jaccard_score = svm_jac
    #     best_kernel_jac = k
    if svm_acc > best_accuracy_svm:
        best_accuracy_svm = svm_acc
        best_kernel = k

print(f"Best Accuracy Score: {round(best_accuracy_svm, 4)} using kernel: {best_kernel}")
# print(f"Best F1 Score: {best_f1_score} using kernel: {best_kernel_f1}")
# print(f"Best Jaccard Similarity Score: {best_jaccard_score} using kernel: {best_kernel_jac}")

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.01).fit(X_train, y_train)
print(lr_model)
lr_pred = lr_model.predict(X_val)
print(lr_pred)

print(f"Logistic Regression Acurracy: {round(accuracy_score(y_val, lr_pred), 4)}")

# %% Evaluation
from sklearn.metrics import f1_score, log_loss


def jaccard_index(predictions, true):
    if len(predictions) == len(true):
        intersect = 0
        for x, y in zip(predictions, true):
            if x == y:
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1


# %% Load test data and preprocess
test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3'
                      '/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv', error_bad_lines=False)
print(test_df.head())

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
                         'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
                         '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False', 'True'], value=[0, 1], inplace=True)
test_X = test_Feature
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)
print(test_X[0:5])

test_y = test_df1['POSTSEASON'].values
print(test_y[0:5])

# %% KNN Evaluation
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_pred = knn_model.predict(test_X)

print(f"KNN Accuracy Score: {round(accuracy_score(test_y, knn_pred), 6)}")
print(f"KNN F1 Score: {round(f1_score(test_y, knn_pred, average='micro'), 6)}")
print(f"KNN Jaccard Similarity Score: {round(jaccard_index(knn_pred, test_y), 6)}")

# %% Decision Tree
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=1).fit(X_train, y_train)
tree_pred = tree_model.predict(test_X)

print(f"Decision Tree Accuracy Score: {round(accuracy_score(test_y, tree_pred), 6)}")
print(f"Decision Tree F1 Score: {round(f1_score(test_y, tree_pred, average='micro'), 6)}")
print(f"Decision Tree Jaccard Similarity Score: {round(jaccard_index(tree_pred, test_y), 6)}")

# %% SVM
svm_model = svm.SVC(kernel='poly').fit(X_train, y_train)
svm_pred = svm_model.predict(test_X)

print(f"SVM Accuracy Score: {round(accuracy_score(test_y, svm_pred), 6)}")
print(f"SVM F1 Score: {round(f1_score(test_y, svm_pred, average='micro'), 6)}")
print(f"SVM Jaccard Similarity Score: {round(jaccard_index(svm_pred, test_y), 6)}")

# %% Logistic Regression
lr_model = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
lr_pred = lr_model.predict(test_X)
yhat_prob = lr_model.predict_proba(test_X)

print(f"Logistic Regression Accuracy Score: {round(accuracy_score(test_y, lr_pred), 6)}")
print(f"Logistic Regression F1 Score: {round(f1_score(test_y, lr_pred, average='micro'), 6)}")
print(f"Logistic Regression Jaccard Similarity Score: {round(jaccard_index(lr_pred, test_y), 6)}")
print(f"Logistic Regression Log Loss: {round(log_loss(test_y, yhat_prob), 6)}")
