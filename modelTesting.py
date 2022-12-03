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
acc_ds_c = []
acc_ds4_c = []
acc_ss_c = []
years = [i for i in range(1970,2022)]

acc_dd_90 = []
acc_ds_90 = []
acc_ds4_90 = []
acc_ss_90 = []
acc_ds_c_90 = []
acc_ds4_c_90 = []
acc_ss_c_90 = []
years_90 = [i for i in range(1990,2022)]

#needed full path for some reason
train_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\data"
train_path_avg = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData"
train_path_comb = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\combinedData"

#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\testData"
#test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\avgData22"

train_data = []
train_data_avg = []
train_data_comb = []

for filename in os.listdir(train_path):
    train_data.append(pd.read_csv(os.path.join(train_path,filename)))

df_train = pd.concat(train_data, ignore_index=True)

for filename in os.listdir(train_path_avg):
    train_data_avg.append(pd.read_csv(os.path.join(train_path_avg,filename)))

df_train_avg = pd.concat(train_data_avg, ignore_index=True)

for filename in os.listdir(train_path_comb):
    train_data_comb.append(pd.read_csv(os.path.join(train_path_comb,filename)))

df_train_comb = pd.concat(train_data_comb, ignore_index=True)

for i in range(1970,2022):
    print(i)

    #game data
    df_test1 = df_train[pd.to_numeric(df_train['year']) == i]
    df_train1 = df_train[pd.to_numeric(df_train['year']) != i]

    X_train = df_train1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_train = df_train1.iloc[:, 34] #labels
    X_test = df_test1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test = df_test1.iloc[:, 34] #labels

    #rolling season average
    df_test_avg1 = df_train_avg[pd.to_numeric(df_train_avg['year']) == i]
    df_train_avg1 = df_train_avg[pd.to_numeric(df_train_avg['year']) != i]

    df_test_avg4 = df_test_avg1[pd.to_numeric(df_test_avg1['week_num']) >= 4]

    X_train_avg = df_train_avg1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_train_avg = df_train_avg1.iloc[:, 34] #labels
    X_test_avg = df_test_avg1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test_avg = df_test_avg1.iloc[:, 34] #labels

    X_test_avg4 = df_test_avg4.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test_avg4 = df_test_avg4.iloc[:, 34] #labels

    #combined season average
    df_test_comb1 = df_train_comb[pd.to_numeric(df_train_comb['year']) == i]
    df_train_comb1 = df_train_comb[pd.to_numeric(df_train_comb['year']) != i]

    df_test_comb4 = df_test_comb1[pd.to_numeric(df_test_comb1['week_num']) >= 4]

    X_train_comb = df_train_comb1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_train_comb = df_train_comb1.iloc[:, 34] #labels
    X_test_comb = df_test_comb1.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test_comb = df_test_comb1.iloc[:, 34] #labels

    X_test_comb4 = df_test_comb4.iloc[:, 6:34] #get everything besides game setting and weather data
    y_test_comb4 = df_test_comb4.iloc[:, 34] #labels

    #need numpy arrays to do accuracy
    y_test = y_test.to_numpy()
    y_test_avg = y_test_avg.to_numpy()
    y_test_avg4 = y_test_avg4.to_numpy()
    y_test_comb = y_test_comb.to_numpy()
    y_test_comb4 = y_test_comb4.to_numpy()
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
    clf.fit(X_train, y_train)
    
    clf_avg = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on rolling season avg
    clf_avg.fit(X_train_avg, y_train_avg)

    clf_comb = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on combined season avg
    clf_comb.fit(X_train_comb, y_train_comb)

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

    #trained on combined avg test on combined avg
    predictions_comb = clf_comb.predict(X_test_comb)
    acc_comb = accuracy_score(y_test_comb, predictions_comb)
    acc_ss_c.append(acc_comb)

    #trained on game test on combined avg
    predictions_real_comb = clf.predict(X_test_comb)
    acc_real_comb = accuracy_score(y_test_comb, predictions_real_comb)
    acc_ds_c.append(acc_real_comb)

    #trained on game test on combined avg >W3
    predictions_real4_comb = clf.predict(X_test_comb4)
    acc_real4_comb = accuracy_score(y_test_comb4, predictions_real4_comb)
    acc_ds4_c.append(acc_real4_comb)

    if i >= 1990:
        #game data
        df_train1 = df_train[pd.to_numeric(df_train['year']) >= 1990]

        df_test1 = df_train1[pd.to_numeric(df_train1['year']) == i]
        df_train1 = df_train1[pd.to_numeric(df_train1['year']) != i]

        X_train = df_train1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_train = df_train1.iloc[:, 34] #labels
        X_test = df_test1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_test = df_test1.iloc[:, 34] #labels

        #rolling season average
        df_train_avg1 = df_train_avg[pd.to_numeric(df_train_avg['year']) >= 1990]

        df_test_avg1 = df_train_avg1[pd.to_numeric(df_train_avg1['year']) == i]
        df_train_avg1 = df_train_avg1[pd.to_numeric(df_train_avg1['year']) != i]

        df_test_avg4 = df_test_avg1[pd.to_numeric(df_test_avg1['week_num']) >= 4]

        X_train_avg = df_train_avg1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_train_avg = df_train_avg1.iloc[:, 34] #labels
        X_test_avg = df_test_avg1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_test_avg = df_test_avg1.iloc[:, 34] #labels

        X_test_avg4 = df_test_avg4.iloc[:, 6:34] #get everything besides game setting and weather data
        y_test_avg4 = df_test_avg4.iloc[:, 34] #labels

        #combined season average
        df_train_comb1 = df_train_comb[pd.to_numeric(df_train_comb['year']) >= 1990]

        df_test_comb1 = df_train_comb1[pd.to_numeric(df_train_comb1['year']) == i]
        df_train_comb1 = df_train_comb1[pd.to_numeric(df_train_comb1['year']) != i]

        df_test_comb4 = df_test_comb1[pd.to_numeric(df_test_comb1['week_num']) >= 4]

        X_train_comb = df_train_comb1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_train_comb = df_train_comb1.iloc[:, 34] #labels
        X_test_comb = df_test_comb1.iloc[:, 6:34] #get everything besides game setting and weather data
        y_test_comb = df_test_comb1.iloc[:, 34] #labels

        X_test_comb4 = df_test_comb4.iloc[:, 6:34] #get everything besides game setting and weather data
        y_test_comb4 = df_test_comb4.iloc[:, 34] #labels

        #need numpy arrays to do accuracy
        y_test = y_test.to_numpy()
        y_test_avg = y_test_avg.to_numpy()
        y_test_avg4 = y_test_avg4.to_numpy()
        y_test_comb = y_test_comb.to_numpy()
        y_test_comb4 = y_test_comb4.to_numpy()
        
        clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on game data
        clf.fit(X_train, y_train)
        
        clf_avg = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on rolling season avg
        clf_avg.fit(X_train_avg, y_train_avg)

        clf_comb = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0) #model trained on combined season avg
        clf_comb.fit(X_train_comb, y_train_comb)

        #trained on game test on game
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        acc_dd_90.append(acc)

        #trained on rolling avg test on rolling avg
        predictions_avg = clf_avg.predict(X_test_avg)
        acc_avg = accuracy_score(y_test_avg, predictions_avg)
        acc_ss_90.append(acc_avg)

        #trained on game test on rolling avg
        predictions_real = clf.predict(X_test_avg)
        acc_real = accuracy_score(y_test_avg, predictions_real)
        acc_ds_90.append(acc_real)

        #trained on game test on rolling avg >W3
        predictions_real4 = clf.predict(X_test_avg4)
        acc_real4 = accuracy_score(y_test_avg4, predictions_real4)
        acc_ds4_90.append(acc_real4)

        #trained on combined avg test on combined avg
        predictions_comb = clf_comb.predict(X_test_comb)
        acc_comb = accuracy_score(y_test_comb, predictions_comb)
        acc_ss_c_90.append(acc_comb)

        #trained on game test on combined avg
        predictions_real_comb = clf.predict(X_test_comb)
        acc_real_comb = accuracy_score(y_test_comb, predictions_real_comb)
        acc_ds_c_90.append(acc_real_comb)

        #trained on game test on combined avg >W3
        predictions_real4_comb = clf.predict(X_test_comb4)
        acc_real4_comb = accuracy_score(y_test_comb4, predictions_real4_comb)
        acc_ds4_c_90.append(acc_real4_comb)

'''
print(acc_dd)
print(acc_ss)
print(acc_ds)
print(acc_ds4)
print(acc_ss_c)
print(acc_ds_c)
print(acc_ds4_c)

print(acc_dd_90)
print(acc_ss_90)
print(acc_ds_90)
print(acc_ds4_90)
print(acc_ss_c_90)
print(acc_ds_c_90)
print(acc_ds4_c_90)
'''
#acc_dd = [0.8520408163265306, 0.8724489795918368, 0.8163265306122449, 0.9081632653061225, 0.9030612244897959, 0.8928571428571429, 0.8571428571428571, 0.9489795918367347, 0.875, 0.9017857142857143, 0.8705357142857143, 0.8705357142857143, 0.8492063492063492, 0.8303571428571429, 0.875, 0.8794642857142857, 0.875, 0.861904761904762, 0.8928571428571429, 0.84375, 0.8348214285714286, 0.8375, 0.8791666666666667, 0.8833333333333333, 0.859375, 0.8860294117647058, 0.8506944444444444, 0.8506944444444444, 0.8576388888888888, 0.8552631578947368, 0.881578947368421, 0.8585526315789473, 0.846875, 0.8625, 0.859375, 0.86875, 0.853125, 0.875, 0.86875, 0.834375, 0.884375, 0.825, 0.85625, 0.84375, 0.8779761904761905, 0.8660714285714286, 0.8125, 0.9107142857142857, 0.8571428571428571, 0.8541666666666666, 0.8154761904761905, 0.8487394957983193]
#acc_ss = [0.5989010989010989, 0.6318681318681318, 0.6428571428571429, 0.6868131868131868, 0.6978021978021978, 0.6978021978021978, 0.6593406593406593, 0.6868131868131868, 0.5761904761904761, 0.6619047619047619, 0.6333333333333333, 0.6476190476190476, 0.6607142857142857, 0.6238095238095238, 0.680952380952381, 0.719047619047619, 0.6666666666666666, 0.6020408163265306, 0.680952380952381, 0.6190476190476191, 0.6428571428571429, 0.6711111111111111, 0.6444444444444445, 0.6488888888888888, 0.6291666666666667, 0.6549019607843137, 0.6777777777777778, 0.6666666666666666, 0.7, 0.6666666666666666, 0.7052631578947368, 0.6666666666666666, 0.63, 0.6766666666666666, 0.6666666666666666, 0.6733333333333333, 0.6633333333333333, 0.67, 0.6666666666666666, 0.6766666666666666, 0.7033333333333334, 0.6266666666666667, 0.65, 0.6766666666666666, 0.6603174603174603, 0.6571428571428571, 0.6476190476190476, 0.6857142857142857, 0.707936507936508, 0.6825396825396826, 0.6476190476190476, 0.6279761904761905]
#acc_ds = [0.6263736263736264, 0.6538461538461539, 0.5989010989010989, 0.6978021978021978, 0.6978021978021978, 0.7087912087912088, 0.6703296703296703, 0.6978021978021978, 0.6142857142857143, 0.6571428571428571, 0.638095238095238, 0.6476190476190476, 0.6785714285714286, 0.6142857142857143, 0.6571428571428571, 0.6714285714285714, 0.6761904761904762, 0.6275510204081632, 0.6904761904761905, 0.6190476190476191, 0.6428571428571429, 0.6888888888888889, 0.6222222222222222, 0.68, 0.625, 0.6352941176470588, 0.6703703703703704, 0.6666666666666666, 0.7185185185185186, 0.6385964912280702, 0.7017543859649122, 0.6666666666666666, 0.6233333333333333, 0.6633333333333333, 0.6533333333333333, 0.67, 0.64, 0.6733333333333333, 0.6866666666666666, 0.65, 0.6566666666666666, 0.6333333333333333, 0.6433333333333333, 0.6633333333333333, 0.6571428571428571, 0.653968253968254, 0.653968253968254, 0.6984126984126984, 0.6761904761904762, 0.6666666666666666, 0.6476190476190476, 0.6190476190476191]
#acc_ds4 = [0.6233766233766234, 0.6493506493506493, 0.6103896103896104, 0.7142857142857143, 0.6948051948051948, 0.7077922077922078, 0.6753246753246753, 0.6753246753246753, 0.6153846153846154, 0.6318681318681318, 0.6263736263736264, 0.6428571428571429, 0.6632653061224489, 0.5989010989010989, 0.6483516483516484, 0.6703296703296703, 0.6648351648351648, 0.6318681318681318, 0.6758241758241759, 0.5934065934065934, 0.6208791208791209, 0.6820512820512821, 0.6122448979591837, 0.685, 0.6009615384615384, 0.6153846153846154, 0.6398305084745762, 0.6822033898305084, 0.7191489361702128, 0.6175298804780877, 0.704, 0.648, 0.6045627376425855, 0.6412213740458015, 0.6551724137931034, 0.6577946768060836, 0.6212121212121212, 0.6730769230769231, 0.6793893129770993, 0.6384615384615384, 0.65, 0.6192307692307693, 0.6423076923076924, 0.6461538461538462, 0.6483516483516484, 0.6336996336996337, 0.6336996336996337, 0.6934306569343066, 0.652014652014652, 0.6410256410256411, 0.6410256410256411, 0.5918367346938775]

#plot stuff
plt.plot(years, acc_dd, label = "Train game, Test game. Avg: " + (str(sum(acc_dd) / len(acc_dd))))

plt.plot(years, acc_ss, label = "Train rolling avg, Test rolling avg. Avg: " + (str(sum(acc_ss) / len(acc_dd))), linestyle='dashed')
plt.plot(years, acc_ds, label = "Train game, Test rolling avg. Avg: " + (str(sum(acc_ds) / len(acc_dd))), linestyle='dashed')
plt.plot(years, acc_ds4, label = "Train game, Test rolling avg >W3. Avg: " + (str(sum(acc_ds4) / len(acc_dd))), linestyle='dashed')

plt.plot(years, acc_ss_c, label = "Train combined avg, Test combined avg. Avg: " + (str(sum(acc_ss_c) / len(acc_dd))))
plt.plot(years, acc_ds_c, label = "Train game, Test combined avg. Avg: " + (str(sum(acc_ds_c) / len(acc_dd))))
plt.plot(years, acc_ds4_c, label = "Train game, Test combined avg >W3. Avg: " + (str(sum(acc_ds4_c) / len(acc_dd))))

plt.legend()
plt.show()
plt.close()

#plot stuff for 90s
plt.plot(years_90, acc_dd_90, label = "Train game, Test game. Avg: " + (str(sum(acc_dd_90) / len(acc_dd_90))))

plt.plot(years_90, acc_ss_90, label = "Train rolling avg, Test rolling avg. Avg: " + (str(sum(acc_ss_90) / len(acc_dd_90))), linestyle='dashed')
plt.plot(years_90, acc_ds_90, label = "Train game, Test rolling avg. Avg: " + (str(sum(acc_ds_90) / len(acc_dd_90))), linestyle='dashed')
plt.plot(years_90, acc_ds4_90, label = "Train game, Test rolling avg >W3. Avg: " + (str(sum(acc_ds4_90) / len(acc_dd_90))), linestyle='dashed')

plt.plot(years_90, acc_ss_c_90, label = "Train combined avg, Test combined avg. Avg: " + (str(sum(acc_ss_c_90) / len(acc_dd_90))))
plt.plot(years_90, acc_ds_c_90, label = "Train game, Test combined avg. Avg: " + (str(sum(acc_ds_c_90) / len(acc_dd_90))))
plt.plot(years_90, acc_ds4_c_90, label = "Train game, Test combined avg >W3. Avg: " + (str(sum(acc_ds4_c_90) / len(acc_dd_90))))

plt.legend()
plt.show()
plt.close()

#save all the accs so we dont have to retrain
file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd', 'wb')
pickle.dump(acc_dd, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss', 'wb')
pickle.dump(acc_ss, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds', 'wb')
pickle.dump(acc_ds, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4', 'wb')
pickle.dump(acc_ds4, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_c', 'wb')
pickle.dump(acc_ss_c, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_c', 'wb')
pickle.dump(acc_ds_c, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4_c', 'wb')
pickle.dump(acc_ds4_c, file)
file.close()

#save all the accs so we dont have to retrain but for 90s
file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_dd_90', 'wb')
pickle.dump(acc_dd_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_90', 'wb')
pickle.dump(acc_ss_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_90', 'wb')
pickle.dump(acc_ds_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4_90', 'wb')
pickle.dump(acc_ds4_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ss_c_90', 'wb')
pickle.dump(acc_ss_c_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds_c_90', 'wb')
pickle.dump(acc_ds_c_90, file)
file.close()

file = open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\results\\acc_ds4_c_90', 'wb')
pickle.dump(acc_ds4_c_90, file)
file.close()