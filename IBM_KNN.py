# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

# %% Read data into dataframe
df = pd.read_csv('teleCust1000t.csv')
df.head()

# %% Data Visualization and Analysis
df['custcat'].value_counts()

df.hist(column='income', bins=50)
plt.show()

# %% Feature Set
df.columns

# to use scikit-learn convert Pandas Dataframe to Numpy array
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
        'reside']].values  # .astype(float)
X[0:5]

y = df['custcat'].values
y[0:5]

# %% Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[:5]

# %% Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(f"Train set: {X_train.shape} {y_train.shape}")
print(f"Test set: {X_test.shape} {y_test.shape}")

# %% Classification
from sklearn.neighbors import KNeighborsClassifier

k = 4
# Train
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)

# Predict
yhat = neigh.predict(X_test)
print(yhat[:5])

# %% Model Evaluation
from sklearn import metrics

print(f'Train set accuracy: {metrics.accuracy_score(y_train, neigh.predict(X_train))}')
print(f'Test set accuracy: {metrics.accuracy_score(y_test, yhat)}')

# %% Find best k
Ks = 10
mean_acc = np.zeros(Ks - 1)
std_acc = np.zeros(Ks - 1)
ConfusionMx = []
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)

    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

mean_acc

# %% Plot Model Accuracy
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
