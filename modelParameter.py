import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import csv

acc_dd = []
acc_ds = []
acc_ds4 = []
acc_ss = []
years = [i for i in range(1970,2022)]

#needed full path for some reason
train_path_avg = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData"
train_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\data"
test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\testData"
test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData22"

train_data = []
train_data_avg = []
test_data = []

count = 0
for filename in os.listdir(train_path):
    train_data.append(pd.read_csv(os.path.join(train_path,filename)))

df_train = pd.concat(train_data, ignore_index=True)

for filename in os.listdir(train_path_avg):
    train_data_avg.append(pd.read_csv(os.path.join(train_path_avg,filename)))

df_train_avg = pd.concat(train_data_avg, ignore_index=True)

f = open("D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results.csv", 'w')
writer = csv.writer(f)

for i in range(1970,1971):
    df_test = df_train[pd.to_numeric(df_train['year']) == i]
    df_train = df_train[pd.to_numeric(df_train['year']) != i]

    X_train = df_train.iloc[:, 6:34] #get everything besides game setting and weather data
    y_train = df_train.iloc[:, 34] #labels
    X_test = df_test.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test = df_test.iloc[:, 34] #labels

    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    df_test_avg = df_train_avg[pd.to_numeric(df_train_avg['year']) == i]
    df_train_avg = df_train_avg[pd.to_numeric(df_train_avg['year']) != i]

    X_train_avg = df_train_avg.iloc[:, 6:34] #get everything besides game setting and weather data
    y_train_avg = df_train_avg.iloc[:, 34] #labels
    X_test_avg = df_test_avg.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test_avg = df_test_avg.iloc[:, 34] #labels

    y_test = y_test.to_numpy()
    y_test_avg = y_test_avg.to_numpy()

    for estimator in range(100, 1100, 100):
        for depth in range(2, 22, 2):
            clf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=0) #this needs to be tuned and all that
            clf.fit(X_train, y_train)
            
            clf_avg = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=0) #this needs to be tuned and all that
            clf_avg.fit(X_train_avg, y_train_avg)

            predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            predictions_avg = clf_avg.predict(X_test_avg)
            acc_avg = accuracy_score(y_test_avg, predictions_avg)

            predictions_real = clf.predict(X_test_avg)
            acc_real = accuracy_score(y_test_avg, predictions_real)

            data = [i, estimator, depth, acc, acc_avg, acc_real]
            print(data)
            writer.writerow(data)

f.close()
