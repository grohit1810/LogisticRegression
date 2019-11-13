# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:47:37 2019

@author: 19233292
"""

import logistic_regression as LogReg
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def SkLearnLogRegTester(data,num_iters = 10):
    col = ["Sample Id","Length","Width","Thickness","Surface Area","Mass","Compactness",
           "Hardness","Shell Top Radius","Water Content","Carbohydrate Content","Variety"]
    dataset = pd.DataFrame(data, columns = col)
    
    dataset = dataset.drop('Sample Id',axis = 1)
    all_acc = 0
    print("SKLearn Logistic Regression implementation : ")
    for i in range(num_iters):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        train, test = train_test_split(dataset, test_size=0.33)
        features_data_train = train.iloc[:,:-1]
        class_data_train = train['Variety']
        features_data_test = test.iloc[:,:-1]
        class_data_test = test['Variety']
        
        log_reg_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000000)
        log_reg_clf.fit(features_data_train, class_data_train)
        acc = log_reg_clf.score(features_data_test,class_data_test)
        print("Accuracy of trial ",i+1," : ",acc)
        all_acc+=acc
    print("Mean Accuracy of 10 trials of SKLearn Logistic Regression implementation : ", all_acc/10)

def MyLogRegTester(data,num_iters = 10):
    X,y = [],[]
    filename = "predictions.txt"
    for line in data :
        line1 = list(map(lambda x: float(x),line[1:len(line)-1]))
        X.append(line1)
        y.append(line[len(line)-1])
        
    X = np.asarray(X)
    X = LogReg.min_max_normalization(X)
    
    data =[]
    for i in range(len(X)):
        dat = []
        dat.append(X[i])
        dat.append(y[i])
        data.append(dat)
    all_acc = 0
    print("My Logistic Regression implementation : ")
    file_exception = False
    try :
        f = open(filename, 'w')
        f.write("#y_predicted,y_actual\n")
        f.close()
    except Exception as e:
        print("Unable to do file operations. Error : ",e)
        file_exception = True
    for i in range(num_iters):
        random.shuffle(data)
        X,y = [],[]
        for dat in data:
            X.append(dat[0])
            y.append(dat[1])
        X = np.asarray(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        classifier = LogReg.LogisticRegression(X_train,y_train)
        classifier.train()
        #classifier.plot_cost_function()
        if not file_exception :
            acc = classifier.calculate_accuracy(X_test,y_test,write_predictions_to_file = True,filename = filename)
        else:
            acc = classifier.calculate_accuracy(X_test,y_test)
        print("Accuracy of trial ",i+1," : ",acc)
        all_acc += acc
        
    print("Mean Accuracy of 10 trials of My Logistic Regression implementation : ", all_acc/10)

if __name__ == "__main__":
    lines =[]
    try:
        file = open("hazelnuts.txt")
        line = file.readline()
        while line : 
            lines.append(line.split())
            line = file.readline()
    except :
        print("Unable to proceed without hazelnuts.txt. Please sure that it is in cwd.")
    finally :
        file.close()
    data = np.transpose(np.array(lines))
    SkLearnLogRegTester(data)
    MyLogRegTester(data)