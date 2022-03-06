from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
# clean data first
df = pd.read_csv('cell2celltrain_Small_6k.csv')
print(df.head(5))
print(df.columns)

# total column count and row count
shape = df.shape
print(shape)

# identify unique values in every column
columns = list(df.columns)

# create numeric_features
numeric_features = []
for col in columns:
    print(f"................start analyazing column {col}...................")
    print(f'whilst there is a unique amount of {shape[0]} rows: ')
    print(f'unique value of column "{col}" is: /n {df[col].unique()}')
    print(f'total count of unique value is: {len(df[col].unique())} ')
    if len(df[col].unique()) == 2:
        numeric_features.append(col)
    print("======================stop================================")

# convert yes or no values into numeric features


def to_numeric(s):
    if s in ["Yes", "Known"]:  # address homeownership feature
        return 1
    elif s == "No":
        return 0
    else:
        return -1


for f in numeric_features:
    df[f] = df[f].apply(to_numeric)

for f in numeric_features:
    print(f'now feature {f} have unique value of {df[f].unique()}')

# catagorize yes and no data
print(" list of yes and no columns: ", numeric_features)


# catagorize one-hot encoding
categorical_features = ['Occupation', 'CreditRating', 'ServiceArea']
df = pd.get_dummies(df, columns=categorical_features, prefix=None)

df= df.replace(to_replace='?', value=np.nan)
df= df.replace(to_replace='Other', value=np.nan)
df = df.dropna(axis='columns')

print('new columns are: ', df.columns)
# train spilt and train data
X = df.drop(labels='Churn', axis=1)
Y = df.Churn
print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


# apply svm as our machine learing model
from sklearn import svm

# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# Print the prediction accuracy in comparsion to actual values
from sklearn import metrics
print(' the accuracy is:', metrics.accuracy_score(y_test, y_pred))