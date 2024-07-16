# -*- coding: utf-8 -*-
#%% import library
import numpy as np

#%% define function

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 오버 플로우 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y