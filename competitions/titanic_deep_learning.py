#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import keras
from keras.models import Sequential
from keras.layers import Dense

def csv(csv):
    return pd.read_csv(csv)

# Drop useless columns
def drop(df, cols):
    return df.drop(columns=cols)


def catCode(column, outputColumn=None):
    if not outputColumn:
        outputColumn = column
    
    outputColumn = pd.Categorical(column)
    outputColumn = outputColumn.codes

    return outputColumn

def fillnaMean(col):
    return col.fillna(col.mean())

def getDummies(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

def rename(df, columns):
    return df.rename(columns=columns)

def firstLetter(col):
    col = col.astype(str).str[0];
    return col

def predictAge(df):
    age = df.drop(columns=['SibSp','Parch'])
    hasAge = age.dropna(subset=['Age'])
    noAge = age[np.isnan(df['Age'])] 
    # Train Age, set age as the label and slice
    X_ageTrain = hasAge.drop(columns=['Age'])
    y_ageTrain = hasAge['Age'];

    X_ageTest = noAge.drop(columns=['Age'])

    lr = Ridge()

    lr.fit(X_ageTrain, y_ageTrain)

    noAge['Age'] = lr.predict(X_ageTest)
    merged = noAge.append(hasAge)
    merged['Age'] = merged['Age'].round(2).copy()
    merged['Age'] = pd.cut(merged['Age'], 8);

    df['Age'] = merged['Age'].cat.codes
    return df

def doTrain(df):
    return df.drop(columns=['Survived'])

def fillTheBlanks(df, col):
    letters = df[col].value_counts(normalize=True)
    nullObjs = df[col].isnull()
    df.loc[nullObjs, col] = np.random.choice(letters.index, size=len(df[nullObjs]), p=letters.values)   
    return df


csvFile = {
    'test': 'test.csv',
    'train': 'train.csv'
}

# print(csvFile)
# open files
td = csv(csvFile['train'])
test_td = csv(csvFile['test'])

# print(td.head())
# drop useless stuff
submission = pd.DataFrame()
submission['PassengerId'] = test_td['PassengerId']

toDrop = ['Ticket', 'PassengerId', 'Name']
td = drop(td, toDrop)
test_td = drop(test_td, toDrop)

c = 'Embarked'
td[c] = catCode(td[c])
test_td[c] = catCode(test_td[c])

td = getDummies(td,['Sex'])
test_td = getDummies(test_td,['Sex'])

td['Fare'] = fillnaMean(td['Fare'])
test_td['Fare'] = fillnaMean(test_td['Fare'])

td['Is_alone'] = td.apply(lambda x: 0 if x['SibSp'] > 0 or x['Parch'] > 0 else 1, axis=1)
test_td['Is_alone'] = test_td.apply(lambda x: 0 if x['SibSp'] > 0 or x['Parch'] > 0 else 1, axis=1)

td['Cabin_letter'] = firstLetter(td['Cabin'])
test_td['Cabin_letter'] = firstLetter(test_td['Cabin'])

td['Cabin_letter'] = td['Cabin_letter'].replace('n', np.nan)
test_td['Cabin_letter'] = test_td['Cabin_letter'].replace('n', np.nan)

td = fillTheBlanks(td, 'Cabin_letter')
test_td = fillTheBlanks(test_td, 'Cabin_letter')

cl = 'Cabin_letter'
td[cl] = catCode(td[cl])
test_td[cl] = catCode(test_td[cl])

td = td.rename(columns={'Sex_male': 'Sex'})
test_td = test_td.rename(columns={'Sex_male': 'Sex'})

td = drop(td, 'Cabin')
test_td = drop(test_td, 'Cabin')

td['Fare_norm'] = abs(td['Fare'] - td['Fare'].mean()) / (td['Fare'].max() - td['Fare'].min())
test_td['Fare_norm'] = abs(test_td['Fare'] - test_td['Fare'].mean()) / (test_td['Fare'].max() - test_td['Fare'].min())

td = drop(td, 'Fare')
test_td = drop(test_td, 'Fare')
td = predictAge(td)
test_td = predictAge(test_td)

td=drop(td, 'Parch')
test_td=drop(test_td, 'Parch')

y_train = td['Survived']

from keras.utils import to_categorical

X_train = doTrain(td)

X_train = to_categorical(X_train)
test_td = to_categorical(test_td)
# print(X_train)
# print(test_td.head(10))
n_cols = X_train.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Loss function: " + model.loss)
print(X_train.shape)
print(test_td.shape)
model.fit(X_train, test_td)
