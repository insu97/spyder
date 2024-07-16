# -*- coding: utf-8 -*-
#%% import library
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from function.DeepLearning import relu, softmax
from model.Drug_model import init_network, predict
#%% data load

df = pd.read_csv("../data/drug200.csv")

print(df.head())

#%% data preprocessing

df = pd.concat([df, pd.get_dummies(df[['Sex','BP','Cholesterol']], dtype=int)], axis=1)
df.drop(columns=['Sex','BP','Cholesterol'], axis=1, inplace=True)

le = LabelEncoder()
df['Drug'] = le.fit_transform(df['Drug'])

print("Drug LabelEncoder : ", le.classes_)

X = df.drop(columns=['Drug'], axis=1)
y = df['Drug']

X = X.values
y = y.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_val = x_train[:40]
x_train = x_train[40:]
y_val = y_train[:40]
y_train = y_train[40:]

#%% DeepLearning

# 황성화 함수
## 은닉층 -> ReLU 사용
## 출력층 -> 다중분류 -> Softmax 사용

network = init_network()

accuracy_cnt = 0
for i in range(len(x_train)):
    y = predict(network, x_train[i])
    p = np.argmax(y)
    if p == y_train[i]:
        accuracy_cnt += 1

print(f"Accuracy: {float(accuracy_cnt) / len(x_train)}")

#%% batch

batch_size = 10
accuracy_cnt = 0

for i in range(0, len(x_train), batch_size):
    x_batch = x_train[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == y_train[i:i+batch_size])

print("Accuracy: "+ str(float(accuracy_cnt) / len(x_train)))




