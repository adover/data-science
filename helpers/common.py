# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.preprocessing import Imputer

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def csv(csv):
    return pd.read_csv(csv)


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


def fillTheBlanks(df, col):
    letters = df[col].value_counts(normalize=True)
    nullObjs = df[col].isnull()
    df.loc[nullObjs, col] = np.random.choice(
        letters.index, size=len(df[nullObjs]), p=letters.values)
    return df


def getDummies(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)


def rename(df, columns):
    return df.rename(columns=columns)


def firstLetter(col):
    col = col.astype(str).str[0]
    return col


def predictMissing(df, col, toDrop):
    cleanedFrame = df.drop(columns=toDrop)
    hasValue = cleanedFrame.dropna(subset=[col])
    noValue = cleanedFrame[np.isnan(df[col])]

    X_train = hasValue.drop(columns=[col])
    y_train = hasValue[col]

    X_test = noValue.drop(columns=[col])

    # TODO: Turn into for loop
    lr = Ridge()

    lr.fit(X_train, y_train)

    noValue[col] = lr.predict(X_test)
    merged = noValue.append(hasValue)
    merged[col] = merged[col].round(2)
    merged[col] = pd.cut(merged[col], 8)

    df[col] = merged[col].cat.codes
    return df


def doTrain(df, target):
    return df.drop(columns=[target])


def doPipeline(p):
    # p = Pipeline([
    #     ("imputer", Imputer(missing_values=0, strategy="mean",axis=0)),
    #     # ('scl', StandardScaler()),
    #     # ('clf',   MLPClassifier(alpha=1))
    #     # ('clf',   AdaBoostClassifier())
    #     # ('clf',  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
    #     # ('clf',  DecisionTreeClassifier(max_depth=5))
    #     # ('clf', GaussianProcessClassifier(1.0 * RBF(1.0)))
    #     # ('clf',  KNeighborsClassifier(1))
    # ])

    return
