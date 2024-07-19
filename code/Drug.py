# -*- coding: utf-8 -*-
#%% import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.Drug_model import simpleNet
from function.DeepLearning import SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
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

#%%%% scaler
select_scaler = input("Scaler 선택 : ['StandardScaler', 'MinMaxScaler', 'Normalizer']")

if select_scaler == 'StandardScaler':
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
elif select_scaler == 'MinMaxScaler':
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
elif select_scaler == 'Normalizer':
    scaler = Normalizer()
    X = scaler.fit_transform(X)

#%%%% train_test_split
# y = pd.get_dummies(y)
y = y.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_val = x_train[:40]
x_train = x_train[40:]
y_val = y_train[:40]
y_train = y_train[40:]

#%% parameter

#%%%% optimizer

select_optimizer = input("optimizer 선택 : ['SGD','Momentum','Nesterov','AdaGrad','RMSprop','Adam']")

if select_optimizer == 'SGD':
    optimizer = SGD()
elif select_optimizer == 'Momentum':
    optimizer = Momentum()
elif select_optimizer == 'Nesterov':
    optimizer = Nesterov()
elif select_optimizer == 'AdaGrad':
    optimizer = AdaGrad()
elif select_optimizer == 'RMSprop':
    optimizer = RMSprop()
elif select_optimizer == 'Adam':
    optimizer = Adam()

#%% mini batch
# 하이퍼파라미터
iters_num= 10000
train_size = x_train.shape[0]
batch_size = 20
learning_rate = 0.01
dropout_ratio = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
     
net = simpleNet(9, [6,6,6,6,6], 5, dropout_ratio=dropout_ratio)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # 기울기 계산
    grad = net.gradient(x_batch, y_batch)
    params = net.params
    optimizer.update(params, grad)
    
    # 학습 경과 기록
    loss = net.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# --- 
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train_loss')
plt.xlabel("epochs")  
plt.ylabel("loss")
plt.show()   