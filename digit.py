# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:19:14 2018

@author: ASHISH
"""


from sklearn import neighbors
import numpy as np


labels = np.load('y.npy')
newx = np.load('x.npy')

train_data = newx[0:49000]
train_label = labels[0:49000]
test_data = newx[50000:55000]
test_label = labels[50000:55000]

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(train_data,train_label)

pred = knn.predict(test_data)

  
total = 0

for i in range(0, pred.shape[0]):
    if pred[i] == test_label[i]:
        total = total +1
        
percentage = (total *1.0)/pred.shape[0] * 100

print percentage
        


