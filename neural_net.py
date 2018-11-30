from os import sys
import pandas as pd
import json

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

from sklearn.neural_network import MLPClassifier


########### READ DATA ##############

# data = sys.argv[1]
data = "./titanic_data/train.csv"
df = pd.read_csv(data, header=0)
# print(df)



########### CLEAN DATA ##############

# parse name
# extract last name and title
parse = df['Name'].str.split(',')
df['LastName'] = parse.str[0]
le = preprocessing.LabelEncoder()
le.fit(df['LastName'])
df['LastName'] = le.transform(df['LastName'])
df['Title'] = parse.str[1].str.split('.').str[0]
le = preprocessing.LabelEncoder()
le.fit(df['Title'])
df['Title'] = le.transform(df['Title'])

# convert sex to number
le = preprocessing.LabelEncoder()
le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])

# group age into child, adult, senior
# fill age missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
# group by age
df['AgeGroup'] = df['Age'].apply(lambda x: 0 if x<=18 else (1 if x<=54 else 2))

# ticket is useless

# cabin is useless

# det. total family size
# combine parch and sibsp
df['FamSize'] = df['Parch'] + df['SibSp']

# fill emabrked missing values
# convert embarked to number
df['Embarked'] = df['Embarked'].fillna('S')
le = preprocessing.LabelEncoder()
le.fit(df['Embarked'])
df['Embarked'] = le.transform(df['Embarked'])



########### SPLIT DATA ##############

X = df[['Pclass', 'LastName', 'Title', 'Sex','AgeGroup','FamSize','Fare','Embarked']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)



########### STANDARDIZE DATA ##############

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



############# PARAMETER TUNING & TRAIN MODEL ###############

tuned_parameters_dt = [{'solver': ['lbfgs'], 'alpha': [.00001, .0005, .0001, .005, .001],
                     'max_iter': [200, 500, 1000], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
                    ]
# clf = GridSearchCV(MLPClassifier(), tuned_parameters_dt, scoring='accuracy', cv=5)
# clf.fit(X_train, y_train)
# best = json.dumps(clf.best_params_)

clf = MLPClassifier(solver='lbfgs', alpha=.00001, max_iter=500, activation='identity')
clf.fit(X_train, y_train)



############# PERFORMANCE EVALUATION ###############

y_true, y_pred = y_test, clf.predict(X_test)
avg_prec = round(precision_score(y_true, y_pred, average='macro'), 2)
avg_recall = round(recall_score(y_true, y_pred, average='macro'), 2)
avg_f1 = round(f1_score(y_true, y_pred, average='macro'), 2)
acc_score = round(accuracy_score(y_true, y_pred), 2)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print("\n\n")
print("NEURAL NETWORK")
print("--------------")
# print("Best Params:\t" + best)
print("Avg F1_Score:\t"+ str(avg_f1))
print("Accuracy Score:\t"+ str(acc_score))
print("\n\n")

plt.title('Neural Network')
plt.plot(false_positive_rate, true_positive_rate,
label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()