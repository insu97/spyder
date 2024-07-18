# -*- coding: utf-8 -*-
#%% import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.Drug_model import simpleNet
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

y = pd.get_dummies(y)

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

net = simpleNet(input_size=9, hidden_size=6, output_size=5)
y = net.predict(x_train)
print(net.accuracy(x_train, y_train))
print(net.loss(x_train, y_train))

#%% mini batch
# 하이퍼파라미터
iters_num= 1000
train_size = x_train.shape[0]
batch_size = 20
learning_rate = 0.5

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
     
net = simpleNet(input_size=9, hidden_size=6, output_size=5)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # 기울기 계산
    grad = net.gradient(x_batch, y_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

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
     