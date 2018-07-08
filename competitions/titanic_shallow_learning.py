# %%
# Helper import
import sys
import os

sys.path.append(os.path.abspath(os.path.join('helpers')))
from common import *

# Lib Import for DS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer

# SKLearn model import
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

# Get CSVs into Pandas dataFrames
csvFile = {
    'train': './competitions/train.csv',
    'test': './competitions/test.csv',
}

td = csv(csvFile['train'])
test_td = csv(csvFile['test'])

# Initialise submission
submission = pd.DataFrame()
submission['PassengerId'] = test_td['PassengerId']

# Get rid of useless columns
toDrop = ['Ticket', 'PassengerId', 'Name']
td = drop(td, toDrop)
test_td = drop(test_td, toDrop)

# Make data categorical
c = 'Embarked'
td[c] = catCode(td[c])
test_td[c] = catCode(test_td[c])

td = getDummies(td, ['Sex'])
test_td = getDummies(test_td, ['Sex'])

# Fill empty values with mean
td['Fare'] = fillnaMean(td['Fare'])
test_td['Fare'] = fillnaMean(test_td['Fare'])

# Create new column based on findings
td['Is_alone'] = td.apply(lambda x: 0 if x['SibSp'] >
                          0 or x['Parch'] > 0 else 1, axis=1)
test_td['Is_alone'] = test_td.apply(
    lambda x: 0 if x['SibSp'] > 0 or x['Parch'] > 0 else 1, axis=1)

# Get useful part of column
td['Cabin_letter'] = firstLetter(td['Cabin'])
test_td['Cabin_letter'] = firstLetter(test_td['Cabin'])

td['Cabin_letter'] = td['Cabin_letter'].replace('n', np.nan)
test_td['Cabin_letter'] = test_td['Cabin_letter'].replace('n', np.nan)

td = fillTheBlanks(td, 'Cabin_letter')
test_td = fillTheBlanks(test_td, 'Cabin_letter')

cl = 'Cabin_letter'
td[cl] = catCode(td[cl])
test_td[cl] = catCode(test_td[cl])

# Tidy up naming
td = td.rename(columns={'Sex_male': 'Sex'})
test_td = test_td.rename(columns={'Sex_male': 'Sex'})

# Get rid of now redundant column
td = drop(td, 'Cabin')
test_td = drop(test_td, 'Cabin')

# Normalise Fare value to be between 0 and 1
td['Fare_norm'] = abs(td['Fare'] - td['Fare'].mean()) / \
    (td['Fare'].max() - td['Fare'].min())
test_td['Fare_norm'] = abs(test_td['Fare'] - test_td['Fare'].mean()) / \
    (test_td['Fare'].max() - test_td['Fare'].min())

# Drop redundant columns
td = drop(td, 'Fare')
test_td = drop(test_td, 'Fare')
td = drop(td, 'Parch')
test_td = drop(test_td, 'Parch')

# Predict NaN age values using regression
unneededAgeFields = ['SibSp']
td = predictMissing(td, 'Age', unneededAgeFields)
test_td = predictMissing(test_td, 'Age', unneededAgeFields)

# Build Pipeline for Model
p = Pipeline([
    ("imputer", Imputer(missing_values=0, strategy="mean", axis=0)),
    ('scl', StandardScaler()),
    # ('clf',   MLPClassifier(alpha=1))
    # ('clf',   AdaBoostClassifier())
    # ('clf',  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
    # ('clf',  DecisionTreeClassifier(max_depth=5))
    # ('clf', GaussianProcessClassifier(1.0 * RBF(1.0)))
    ('clf',  KNeighborsClassifier(1))
])

# Create training feature
y_train = td['Survived']
X_train = doTrain(td, 'Survived')

# Fit to data to model
p.fit(X_train, y_train)

# Generate prediction
y_pred = p.predict(test_td)

# Build submission
submission = pd.DataFrame({
    "PassengerId": submission["PassengerId"],
    "Survived": y_pred
})
submission.to_csv('./submission_2.csv', index=False)

# Get accuracy of prediction
acc_svc = round(p.score(X_train, y_train) * 100, 2)
print(acc_svc)
