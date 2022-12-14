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
acc_ds12 = []
acc_ss = []
years = [i for i in range(1970,2022)]

#needed full path for some reason
train_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\normData"
train_path_avg = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\combinedNormData"

#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\testData"
#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData22"


train_data = []
train_data_avg = []

for filename in os.listdir(train_path):
    train_data.append(pd.read_csv(os.path.join(train_path,filename)))

df_train = pd.concat(train_data, ignore_index=True)

for filename in os.listdir(train_path_avg):
    train_data_avg.append(pd.read_csv(os.path.join(train_path_avg,filename)))

df_train_avg = pd.concat(train_data_avg, ignore_index=True)

'''
array = ['purple', 'purple', 'purple', 'red', 'blue', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'purple', 'purple', 'purple', 'blue', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
array.reverse()
colors = np.array(array)
df = pd.read_csv("D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData\\atl.csv")

X_train = df_train.iloc[:, 6:34] #get everything besides game setting and weather data
y_train = df_train.iloc[:, 34] #labels

clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
clf.fit(X_train, y_train)

#clf.feature_importances_
sorted_idx = clf.feature_importances_.argsort()
names = df.columns[6:34]
plt.rc('ytick', labelsize=24)
plt.barh(names[sorted_idx], clf.feature_importances_[sorted_idx], color=colors[sorted_idx])
plt.show()
plt.close()

X_train = df_train_avg.iloc[:, 6:34] #get everything besides game setting and weather data
y_train = df_train_avg.iloc[:, 34] #labels

clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
clf.fit(X_train, y_train)

#clf.feature_importances_
sorted_idx = clf.feature_importances_.argsort()
names = df.columns[6:34]
plt.rc('ytick', labelsize=24)
plt.barh(names[sorted_idx], clf.feature_importances_[sorted_idx], color= colors[sorted_idx])
plt.show()
plt.close()
'''

for i in range(1970,2022):
    print(i)

    #game data
    df_test1 = df_train[pd.to_numeric(df_train['year']) == i]
    df_train1 = df_train[pd.to_numeric(df_train['year']) != i]

    X_train = df_train1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_train = df_train1.iloc[:, 34] #labels
    X_test = df_test1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_test = df_test1.iloc[:, 34] #labels

    #rolling season average
    df_test_avg1 = df_train_avg[pd.to_numeric(df_train_avg['year']) == i]
    df_train_avg1 = df_train_avg[pd.to_numeric(df_train_avg['year']) != i]

    df_test_avg4 = df_test_avg1[pd.to_numeric(df_test_avg1['week_num']) >= 4]
    df_test_avg12 = df_test_avg1[pd.to_numeric(df_test_avg1['week_num']) >= 12]

    X_train_avg = df_train_avg1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_train_avg = df_train_avg1.iloc[:, 34] #labels
    X_test_avg = df_test_avg1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_test_avg = df_test_avg1.iloc[:, 34] #labels

    X_test_avg4 = df_test_avg4.iloc[:, 4:34] #get everything besides game setting and weather data
    y_test_avg4 = df_test_avg4.iloc[:, 34] #labels

    X_test_avg12 = df_test_avg12.iloc[:, 4:34] #get everything besides game setting and weather data
    y_test_avg12 = df_test_avg12.iloc[:, 34] #labels

    #need numpy arrays to do accuracy
    y_test = y_test.to_numpy()
    y_test_avg = y_test_avg.to_numpy()
    y_test_avg4 = y_test_avg4.to_numpy()
    y_test_avg12 = y_test_avg12.to_numpy()
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
    clf.fit(X_train, y_train)
    
    clf_avg = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on rolling season avg
    clf_avg.fit(X_train_avg, y_train_avg)

    #trained on game test on game
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    acc_dd.append(acc)

    #trained on rolling avg test on rolling avg
    predictions_avg = clf_avg.predict(X_test_avg)
    acc_avg = accuracy_score(y_test_avg, predictions_avg)
    acc_ss.append(acc_avg)

    #trained on game test on rolling avg
    predictions_real = clf.predict(X_test_avg)
    acc_real = accuracy_score(y_test_avg, predictions_real)
    acc_ds.append(acc_real)

    #trained on game test on rolling avg >W3
    predictions_real4 = clf.predict(X_test_avg4)
    acc_real4 = accuracy_score(y_test_avg4, predictions_real4)
    acc_ds4.append(acc_real4)

    #trained on game test on rolling avg >W11
    predictions_real12 = clf.predict(X_test_avg12)
    acc_real12 = accuracy_score(y_test_avg12, predictions_real12)
    acc_ds12.append(acc_real12)

#plot stuff
plt.plot(years, acc_dd, label = "Train game, Test game. Avg: " + (str(sum(acc_dd) / len(acc_dd))))

plt.plot(years, acc_ss, label = "Train combined avg, Test combined avg. Avg: " + (str(sum(acc_ss) / len(acc_dd))))
plt.plot(years, acc_ds, label = "Train game, Test combined avg. Avg: " + (str(sum(acc_ds) / len(acc_dd))))
plt.plot(years, acc_ds4, label = "Train game, Test combined avg >W3. Avg: " + (str(sum(acc_ds4) / len(acc_dd))))
plt.plot(years, acc_ds12, label = "Train game, Test combined avg >W11. Avg: " + (str(sum(acc_ds12) / len(acc_dd))))

plt.legend()
plt.show()
plt.close()

#save all the accs so we dont have to retrain
file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd_n_norm', 'wb')
pickle.dump(acc_dd, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_n_norm', 'wb')
pickle.dump(acc_ss, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_n_norm', 'wb')
pickle.dump(acc_ds, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4_n_norm', 'wb')
pickle.dump(acc_ds4, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds12_n_norm', 'wb')
pickle.dump(acc_ds12, file)
file.close()