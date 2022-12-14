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
acc_ss = []

#needed full path for some reason
train_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\data"
train_path_avg = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\combinedData"

#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\testData"
#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData22"

'''
train_data = []
train_data_avg = []

for filename in os.listdir(train_path):
    train_data.append(pd.read_csv(os.path.join(train_path,filename)))

df_train = pd.concat(train_data, ignore_index=True)

for filename in os.listdir(train_path_avg):
    train_data_avg.append(pd.read_csv(os.path.join(train_path_avg,filename)))

df_train_avg = pd.concat(train_data_avg, ignore_index=True)
'''
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
'''
for i in range(1,19):
    print(i)

    #game data
    df_test1 = df_train[pd.to_numeric(df_train['week_num']) == i]
    df_train1 = df_train[pd.to_numeric(df_train['week_num']) != i]

    X_train = df_train1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_train = df_train1.iloc[:, 34] #labels
    X_test = df_test1.iloc[:, 4:34] #get everything besides game setting and weather data
    y_test = df_test1.iloc[:, 34] #labels

    #rolling season average
    if(i > 1):
        df_test_avg1 = df_train_avg[pd.to_numeric(df_train_avg['week_num']) == i]
        df_train_avg1 = df_train_avg[pd.to_numeric(df_train_avg['week_num']) != i]

        X_train_avg = df_train_avg1.iloc[:, 4:34] #get everything besides game setting and weather data
        y_train_avg = df_train_avg1.iloc[:, 34] #labels
        X_test_avg = df_test_avg1.iloc[:, 4:34] #get everything besides game setting and weather data
        y_test_avg = df_test_avg1.iloc[:, 34] #labels

        y_test_avg = y_test_avg.to_numpy()

        clf_avg = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on rolling season avg
        clf_avg.fit(X_train_avg, y_train_avg)

        #trained on rolling avg test on rolling avg
        predictions_avg = clf_avg.predict(X_test_avg)
        acc_avg = accuracy_score(y_test_avg, predictions_avg)
        acc_ss.append(acc_avg)

    #need numpy arrays to do accuracy
    y_test = y_test.to_numpy()
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
    clf.fit(X_train, y_train)

    #trained on game test on game
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    acc_dd.append(acc)

    if(i > 1):
        #trained on game test on rolling avg
        predictions_real = clf.predict(X_test_avg)
        acc_real = accuracy_score(y_test_avg, predictions_real)
        acc_ds.append(acc_real)

#plot stuff
plt.plot([i for i in range(1,19)], acc_dd, label = "Train game, Test game. Avg: " + (str(sum(acc_dd) / len(acc_dd))))

plt.plot([i for i in range(2,19)], acc_ss, label = "Train combined avg, Test combined avg. Avg: " + (str(sum(acc_ss) / len(acc_dd))))
plt.plot([i for i in range(2,19)], acc_ds, label = "Train game, Test combined avg. Avg: " + (str(sum(acc_ds) / len(acc_dd))))

plt.legend()
plt.show()
plt.close()

#save all the accs so we dont have to retrain
file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd_n_w', 'wb')
pickle.dump(acc_dd, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_n_w', 'wb')
pickle.dump(acc_ss, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_n_w', 'wb')
pickle.dump(acc_ds, file)
file.close()
'''

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd_n_w', 'rb') as f:
    acc_dd = pickle.load(f)
acc_dd = acc_dd[:-1]

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_n_w', 'rb') as f:
    acc_ss = pickle.load(f)
acc_ss = acc_ss[:-1]

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_n_w', 'rb') as f:
    acc_ds = pickle.load(f)
acc_ds = acc_ds[:-1]
'''

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd_n_norm', 'rb') as f:
    acc_dd = pickle.load(f)

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_n_norm', 'rb') as f:
    acc_ss = pickle.load(f)

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_n_norm', 'rb') as f:
    acc_ds = pickle.load(f)

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4_n_norm', 'rb') as f:
    acc_ds4 = pickle.load(f)

with open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds12_n_norm', 'rb') as f:
    acc_ds12 = pickle.load(f)
'''

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.plot([i for i in range(1,18)], acc_dd, label = "Train game, Test game. Avg: " + (str(round(sum(acc_dd) / len(acc_dd),3))))

plt.plot([i for i in range(2,18)], acc_ss, label = "Train combined avg, Test combined avg. Avg: " + (str(round(sum(acc_ss) / len(acc_dd),3))))
plt.plot([i for i in range(2,18)], acc_ds, label = "Train game, Test combined avg. Avg: " + (str(round(sum(acc_ds) / len(acc_dd),3))))

plt.legend()
plt.xticks(np.arange(1, 17+1, 1.0))
plt.xlabel("Week in Season")
plt.ylabel("Accuracy")
plt.title("Leave-One-Out Cross Validation on Week", fontdict={'fontsize': 34})
plt.show()
'''

plt.plot([i for i in range(1970,2022)], acc_dd, label = "Train game, Test game. Avg: " + (str(round(sum(acc_dd) / len(acc_dd),3))))

plt.plot([i for i in range(1970,2022)], acc_ss, label = "Train combined avg, Test combined avg. Avg: " + (str(round(sum(acc_ss) / len(acc_dd),3))))
plt.plot([i for i in range(1970,2022)], acc_ds, label = "Train game, Test combined avg. Avg: " + (str(round(sum(acc_ds) / len(acc_dd),3))))
#plt.plot([i for i in range(1970,2022)], acc_ds4, label = "Train game, Test combined avg >W3. Avg: " + (str(round(sum(acc_ds4) / len(acc_dd),3))))
plt.plot([i for i in range(1970,2022)], acc_ds12, label = "Train game, Test combined avg >W11. Avg: " + (str(round(sum(acc_ds12) / len(acc_dd),3))))

plt.legend()
plt.xticks(np.arange(1970, 2021+1, 5.0))
plt.xlabel("Year")
plt.ylabel("Accuracy")
plt.title("Leave-One-Out Cross Validation on Year - Normalized", fontdict={'fontsize': 34})
plt.show()
'''