import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

#needed full path for some reason
train_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\data"
test_path = "D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\\testData"

train_data = []
test_data = []

count = 0
for filename in os.listdir(train_path):
    train_data.append(pd.read_csv(os.path.join(train_path,filename)))

df_train = pd.concat(train_data, ignore_index=True)


X = df_train.iloc[:, 6:34] #get everything besides game setting and weather data
y = df_train.iloc[:, 34] #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0) #this needs to be tuned and all that
clf.fit(X_train, y_train)
#to save the model
pickle.dump(clf, open('D:\School\Fall 22\Pattern Analysis\CSCE666_NFL\RandomForestClassifier.sav','wb'))

y_test = y_test.to_numpy()
predictions = clf.predict(X_test)
#for c in range(10):
#    print(predictions[c], y_test[c])

acc = accuracy_score(y_test, predictions)
print("accuracy is:", acc)

cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


for filename in os.listdir(test_path):
    test_data.append(pd.read_csv(os.path.join(test_path,filename)))

df_test = pd.concat(test_data, ignore_index=True)
#display(df_test)