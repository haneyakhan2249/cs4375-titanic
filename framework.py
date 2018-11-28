# ************* FRAMEWORK / USELESS STUFF **********************

from os import sys
import pandas as pd
import json

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


########### READ DATA ##############

# data = sys.argv[1]
train_data = "./titanic_data/train.csv"
train_df = pd.read_csv(train_data, header=0)
# print(train_df)

# data = sys.argv[1]
test_data = "./titanic_data/test.csv"
test_df = pd.read_csv(test_data, header=0)
# print(test_df)


########### CLEAN DATA ##############

# remove name
# convert sex to number
le = preprocessing.LabelEncoder()
le.fit(train_df['Sex'])
train_df['Sex'] = le.transform(train_df['Sex'])
# fill age missing values
train_df['Age'] = train_df['Age'].fillna('0')
# convert ticket to number
le = preprocessing.LabelEncoder()
le.fit(train_df['Ticket'])
train_df['Ticket'] = le.transform(train_df['Ticket'])
# remove cabin for now
# convert embarked to number
le = preprocessing.LabelEncoder()
le.fit(train_df['Embarked'].fillna('0'))
train_df['Embarked'] = le.transform(train_df['Embarked'].fillna('0'))

# print(train_df)


######### COLLECT STATS ON DATA ############## 

# count = 0
# for index, row in df.iterrows():
#     if(row['Survived'] == 1):
#         count+=1
# print(count)

# count = 0
# for index, row in df.iterrows():
#     if(row['Survived'] == 0):
#         count+=1
# print(count)


######### SPLIT TRAIN DATA FOR TESTING ############## 

X = train_df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']].values
y = train_df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)



############# DECISION TREE PARAMETER TUNING ###############

tuned_parameters_dt = [{'max_depth': [1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     'max_features': [2, 9]}
                    ]
clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters_dt, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)
best = json.dumps(clf.best_params_)

y_true, y_pred = y_test, clf.predict(X_test)
avg_prec = round(precision_score(y_true, y_pred, average='macro'), 2)
avg_recall = round(recall_score(y_true, y_pred, average='macro'), 2)
avg_f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
acc_score = round(accuracy_score(y_true, y_pred), 2)

print("\n\n")
print("DECISION TREE")
print("--------------")
print("Best Params:\t" + best)
print("Avg Precision:\t"+ str(avg_prec))
print("Avg Recall:\t"+ str(avg_recall))
print("Avg F1_Score:\t"+ str(avg_f1))
print("Accuracy Score:\t"+ str(acc_score))
print("\n\n")



############# NEURAL NETWORK PARAMETER TUNING ###############

tuned_parameters_dt = [{'solver': ['lbfgs'], 'alpha': [.0005, .0001, .005, .001, .01],
                     'max_iter': [500, 1000], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
                    ]
clf = GridSearchCV(MLPClassifier(), tuned_parameters_dt, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)
best = json.dumps(clf.best_params_)

y_true, y_pred = y_test, clf.predict(X_test)
avg_prec = round(precision_score(y_true, y_pred, average='macro'), 2)
avg_recall = round(recall_score(y_true, y_pred, average='macro'), 2)
avg_f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
acc_score = round(accuracy_score(y_true, y_pred), 2)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("NEURAL NETWORK")
print("--------------")
print("Best Params:\t" + best)
print("Avg Precision:\t"+ str(avg_prec))
print("Avg Recall:\t"+ str(avg_recall))
print("Avg F1_Score:\t"+ str(avg_f1))
print("Accuracy Score:\t"+ str(acc_score))
print("\n\n")