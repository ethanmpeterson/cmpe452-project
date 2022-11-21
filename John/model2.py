#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

def main():
    attempt = 2 # [0, 3]

    data_dict = load_data_dict()

    if attempt == 0:
        model = svm.SVC()
        model.fit(data_dict['x_train'], data_dict['y_train'])
    elif attempt == 1:
        model = svm.SVC()
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        data_dict['x_train_reduced'], data_dict['y_train_reduced'] = rus.fit_resample(data_dict['x_train'], data_dict['y_train'])
        model.fit(data_dict['x_train_reduced'], data_dict['y_train_reduced'])
    elif attempt == 2:
        model = svm.SVC(kernel='linear')
        model.fit(data_dict['x_train'], data_dict['y_train'])
    elif attempt == 3:
        model = svm.SVC(kernel='sigmoid')
        model.fit(data_dict['x_train'], data_dict['y_train'])

    data_dict['y_pred'] = model.predict(data_dict['x_val'])
    evaluator(data_dict['y_val'], data_dict['y_pred'])


# last column is the class to balance
def balance(set):
    pass

'''
splits Train.csv into x and y for training and validation, each.
optionally loads Test.csv into x_test

test_preprocess is a functor which is called on the training data before being split up
'''
def load_data_dict(validation_portion = 1/3, get_test = False):
    script_dir = os.path.dirname(__file__)
    train_csv_path = os.path.join(script_dir, "..", "data", "Train.csv")
    train_csv = np.genfromtxt(train_csv_path, delimiter=',', skip_header=True)
    if get_test:
        test_csv_path = os.path.join(script_dir, "..", "data", "Test.csv")
        test_csv = np.genfromtxt(test_csv_path, delimiter=',', skip_header=True)

    data_dict = {}
    x_train = train_csv[:,:-1]
    y_train = train_csv[:,-1:].transpose()[0]
    data_dict['x_train'], data_dict['x_val'] = train_test_split(x_train, test_size=validation_portion)
    data_dict['y_train'], data_dict['y_val'] = train_test_split(y_train, test_size=validation_portion)
    if get_test:
        data_dict['x_test'] = test_csv
    
    return data_dict

def evaluator(y_test, y_pred, n=2, all=False):
    # both are arrays with values from 0-n
    confusion_matrix = np.zeros((n, n), dtype=np.int)
    for test, pred in zip(y_test, y_pred):
        # matrix is row major, and predicted values are each column
        confusion_matrix[int(test)][int(pred)] += 1
    print("confusion matrix:\n" + str(confusion_matrix))

    if all:
        accuracy = confusion_matrix.trace() / confusion_matrix.sum() # correct / all
        print("accuracy: %f" % accuracy)

        # true positive / total positive
        precision = np.empty((n))
        for i in range(n):
            sum_positive = 0
            for j in range(n):
                sum_positive += confusion_matrix[j][i]
            precision[i] = confusion_matrix[i][i] / sum_positive
        print("precision:\n" + str(precision))

        # true positive / total positive
        recall = np.empty((n))
        for i in range(n):
            recall[i] = confusion_matrix[i][i] / confusion_matrix[i].sum()
        print("recall:\n" + str(recall))

        f1 = np.empty((n))
        for i in range(n):
            f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        print("f1:\n" + str(f1))

if __name__ == "__main__":
    main()
