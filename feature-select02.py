from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# clean data first
train = pd.read_csv('cell2celltrain_Small_6k.csv')

# shape of the data
col_num, row_num = train.shape

print(train.describe())
time.sleep(1)

# label end churn data as either 0 or 1
train['Churn'].replace(to_replace='Yes', value=1, inplace=True)
train['Churn'].replace(to_replace='No',  value=0, inplace=True)

# convert all dataset value that should be float from string
def check_numeric(val):
    return val.replace('.','',1).isnumeric()

for col in train.columns:
    if train[col].dtype == 'object':
        max_val_1, max_val_2 = train[col].value_counts()[:2].index.tolist()
        if check_numeric(max_val_1) or check_numeric(max_val_2):
            train[col] = pd.to_numeric(train[col], errors='coerce') # will convert any ? into nan
            print(f'{col} contains numeric value {max_val_1} with max occurance')


def describe_unique_catagorical_data(train):
    # categoric features
    for i in train.columns:
        if train[i].dtype == 'object':
            print(pd.DataFrame(train[i].value_counts()))

def describe_unique_data(train):
    for i in train.columns:
            print(pd.DataFrame(train[i].value_counts()))

def describe_total_unique_data(df):
    for col in df.columns:
        total_unique_values = len(df[col].unique())
        print(f'column {col} has {total_unique_values} unique values in total')
        print(f'these values are {df[col].unique()}')

def debug_nan(df):
    for i in df.columns:
        print(f'in column {i} there is a total of {df[i].isna().sum()} null values')

# handling missing values in each columns
unknown_values = ['?', 'Unknown', 'Other']
for i in unknown_values:
    train = train.replace(i, np.nan)

# initial data exploration wshow occupation mostly fails into 'Other'
# this means that it can be ignored
# initial data analysis also shown that handset price are mostly unknown, therefore it will not be considered either
# this goes for prizmcode as well
print('columns contain null values are: ', train.columns[train.isnull().any()])


def fliter_large_missing_values(df):
    print("this function prints out the total number of null values")
    col_to_drop = []
    FLITER_VAL = 30  # amount we allow data to be missing in percentage
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            missing_percent = (df[col].isnull().sum() / col_num) * 100
            print(col, df[col].isnull().sum(),
                  '{:.2f}%'.format(missing_percent))
            if missing_percent > FLITER_VAL:
                col_to_drop.append(col)
    return col_to_drop


# drop all the large missing values
col_to_drop = fliter_large_missing_values(train)
print(f'columns with large amount of missing data are {col_to_drop}')
print('dropping the columns now')
df = train.drop(col_to_drop, axis=1)

# put dfata into numerical and catagorical
# initial analysis show dataset only have float or object type
categoricals = list()
numericals = list()
for x in df.columns:
    if df[x].dtype == 'object':
        categoricals.append(x)
    else:
        numericals.append(x)

# for numerical data we will analyze correlation
# we use pearson's r for correlation analysis; the result is to filter out feature that are way to correlated to each other
# it will form a matrix by comparing each column to each other to get the correlation value
# we loop through the dataframe to filter out columns with big correlation
correlated_features = set()
correlation_matrix = df[numericals].corr()
R_VALUE = 0.7
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > R_VALUE:
            colname1 = correlation_matrix.columns[i]
            colname2 = correlation_matrix.columns[j]
            # print (correlation_matrix.columns[i] + ' and ' + correlation_matrix.columns[j])
            if colname1 != 'Churn' and colname2 != 'Churn':
                if abs(correlation_matrix['Churn'][colname1]) > abs(correlation_matrix['Churn'][colname2]):
                    correlated_features.add(colname2)
                else:
                    correlated_features.add(colname1)
df.drop(correlated_features, axis=1, inplace=True)


# catagorize data into feature and results
# we will first use one-hot encoding to convert catagorical data into numerical data
def to_numeric(s):
    if s == 'Yes': 
        return 1
    elif s == "No":
        return 0
    else:
        return -1

for col in df.columns:
    if df[col].dtype == 'object':
        if len(df[col].unique()) <= 2:
            df[col] = df[col].apply(to_numeric)



# eliminate categorical features that have too many unique values
# that is more than 5% of the values are unique, i.e. this case we will have to drop the column
for feature in categoricals:
    if len(df[feature].unique()) / df.shape[0] >= 0.05:
        print('big feature is: ', feature)
        df = df.drop([feature], axis=1)
print(f'cleaned all big features, now the shape is {df.shape}')

# drop unrelated data manually
df = df.drop(['NonUSTravel', 'TruckOwner', 'RVOwner', 'OwnsComputer', 'ChildrenInHH', 'OwnsMotorcycle', 'UniqueSubs'], axis=1)
df = df.drop(['NewCellphoneUser', 'NotNewCellphoneUser'], axis=1)
df = df.drop(['HandsetRefurbished', 'HandsetWebCapable', 'HandsetModels', 'CurrentEquipmentDays'], axis=1)
# encoding catagorical data
df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies.columns)

# Impute missing values
# use this function to debug any missing values: debug_nan(df_dummies)

from sklearn.impute import SimpleImputer
from numpy import isnan
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# transform the dataset
df_cleaned = pd.DataFrame(imputer.fit_transform(df_dummies))
# count the number of NaN values in each column
df_cleaned.columns=df_dummies.columns
df_cleaned.index=df_dummies.index


X = df_cleaned.drop(['Churn'], axis=1)
Y = df_cleaned['Churn']

print(X.info(memory_usage = "deep"))

# preprocessing data to make it standardized
# minmaxscaler is best for data that assume no normal distribution is found is data
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# some descriptive data for our final dataset
print('the head of dataframe is: ', X.head())
print(X.shape)
print(X.describe())

# train spilt and train data. 40 percent for training 60 percent for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# apply svm as our machine learing model
from sklearn import svm

# Create a svm Classifier
clf = svm.SVC(kernel='poly') # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# Print the prediction accuracy in comparsion to actual values
from sklearn import metrics
print(' the accuracy is:', metrics.accuracy_score(y_test, y_pred))
