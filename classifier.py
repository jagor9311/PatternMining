#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:05:11 2020

@author: rli25
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score, recall_score
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
 
import RTPmining

#Data = list of MSS, Omega = candidate list of lists, g = gap
def create_binary_matrix(Data,Omega,g,g_int):
    binaryMatrix = np.zeros((len(Data),len(Omega)))
    for i in range(0,len(Data)):
        for j in range(0,len(Omega)):
                present, endT = RTPmining.recent_temporal_pattern(Data[i],Omega[j],g,g_int)
                if(present):
                    binaryMatrix[i,j] = 1
                else:
                    binaryMatrix[i,j]= 0
    return binaryMatrix

def HMM_binary_matrix(Data,Omega_shock,Omega_nonshock,g,g_int,interval_length):
    matrix_set = []
    for i in range(0,len(Data)):
        binary_Matrix1 = np.zeros((len(Omega_shock),int(g/interval_length)))
        binary_Matrix2 = np.zeros((len(Omega_nonshock),int(g/interval_length)))
        for j in range(0,len(Omega_shock)):
            present, endT = RTPmining.recent_temporal_pattern(Data[i],Omega_shock[j],g,g_int)
            if(present):
                for l in range(0,int(g/interval_length)):
                    is_lRTP = RTPmining.l_recent_temporal_pattern(Data[i],Omega_shock[j],g,g_int,l,interval_length)
                    if(is_lRTP):
                        binary_Matrix1[j,l] = 1
                    else:
                        binary_Matrix1[j,l] = 0
        for j_non in range(0,len(Omega_nonshock)):
            present, endT = RTPmining.recent_temporal_pattern(Data[i],Omega_nonshock[j_non],g,g_int)
            if(present):
                for l_non in range(0,int(g/interval_length)):
                    is_lRTP = RTPmining.l_recent_temporal_pattern(Data[i],Omega_nonshock[j_non],g,g_int,l_non,interval_length)
                    if(is_lRTP):
                        binary_Matrix2[j_non,l_non] = 1
                    else:
                        binary_Matrix2[j_non,l_non] = 0
        finalmatrix = np.concatenate((binary_Matrix1,binary_Matrix2),axis=0)
        matrix_set.append(finalmatrix)
        
    return matrix_set

def new_HMM_binary_matrix(Data,changing_shock_set,changing_nonshock_set,g,g_int,interval_length):
    matrix_set = []
    test_array = []
    test_val_array = []
    for l in range(0,int(g/interval_length)):
        finalArraylist = []
        Omega_shock = changing_shock_set[l]
        Omega_nonshock = changing_nonshock_set[l]
        #Omega_shock = changing_shock_set
        #Omega_nonshock = changing_nonshock_set
        for i in range(0,len(Data)):
            binary_Array1 = np.zeros(len(Omega_shock))
            binary_Array2 = np.zeros(len(Omega_nonshock))
            for j in range(0,len(Omega_shock)):
                present, endT = RTPmining.recent_temporal_pattern(Data[i],Omega_shock[j],g,g_int)
                if (present):
                    is_lRTP, val = RTPmining.l_recent_temporal_pattern(Data[i],Omega_shock[j],g,g_int,l,interval_length)
                    test_val = [present, is_lRTP]
                    test_val_array.append(val)
                    if (is_lRTP):
                        binary_Array1[j] = 1
                    else:
                        binary_Array1[j] = 0
                else:
                    test_val = [present]
                test_array.append(test_val)
            for j_non in range(0, len(Omega_nonshock)):
                present, endT = RTPmining.recent_temporal_pattern(Data[i],Omega_nonshock[j_non],g,g_int)
                if (present):
                    is_lRTP, val= RTPmining.l_recent_temporal_pattern(Data[i],Omega_nonshock[j_non],g,g_int,l,interval_length)
                    test_val_array.append(val)
                    if (is_lRTP):
                        binary_Array2[j_non] = 1
                    else:
                        binary_Array2[j_non] = 0
            finalArray = np.concatenate((binary_Array1,binary_Array2),axis=0)
            finalArraylist.append(finalArray)
        matrix_set.append(finalArraylist)
    
    new_matrix_set = []
    for i in range(0,len(matrix_set[0])):
        binaryMatrix = np.zeros((len(matrix_set[0][0]),len(matrix_set)))
        for j in range(0,len(matrix_set)):
            binaryMatrix[:,j] = matrix_set[j][i]
        new_matrix_set.append(binaryMatrix)
    
    return new_matrix_set, test_val_array
            

def get_tuple_each_time(Data,pattern_set,g,g_int,interval_length):
    changing_frequent_pattern_ls = []
    for l in range(0,int(g/interval_length)):
        pattern_list = []
        count_list = []
        for j in range(0,len(pattern_set)):
            count = 0
            for i in range(0,len(Data)):
                present, endT = RTPmining.recent_temporal_pattern(Data[i],pattern_set[j],g,g_int)
                if(present):
                    is_lRTP, val= RTPmining.l_recent_temporal_pattern(Data[i],pattern_set[j],g,g_int,l,interval_length)
                    if(is_lRTP):
                        count = count + 1
            if count > 0:
                pattern_list.append(pattern_set[j])
                count_list.append(count)
        changing_frequent_pattern_ls.append(list(zip(pattern_list,count_list)))
    
    return changing_frequent_pattern_ls
    
        
        
            
def learn_svm(trainData,trainLabels,testData,testLabels):
    print ("SVM Learning........")
    clf = svm.SVC(kernel='rbf', probability=True)
    param_grid = {"gamma" : [1, 1e-1, 1e-2, 1e-3], "C" : [1, 10, 100]}
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, n_jobs=5)
    clf.fit(trainData, trainLabels)
    print (clf.best_params_)
    # clf = clf.fit(trainData, trainLabels)
    train_predicted = clf.predict(trainData)
    accuracy = accuracy_score(trainLabels, train_predicted)
    print ("training accuracy:", accuracy)
    test_predicted = clf.predict(testData)
    accuracy = accuracy_score(testLabels, test_predicted)
    print ("test accuracy:", accuracy)
    # #print "Accuracy:    ", accuracy
    # precision, recall, f_measure, _ = precision_recall_fscore_support(testLabels, predicted, pos_label=1, average='weighted')
    conf_matrix = metrics.confusion_matrix(testLabels,test_predicted)
    print (conf_matrix)
    print(classification_report(testLabels,test_predicted,digits=4))
    # print "precision:    ", precision
    # print "recall:        ", recall
    # print "F-measure:    ", f_measure
    auc = roc_auc_score(testLabels,test_predicted)
    print ("ROC under AUC:", auc)
    
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    #print(N_train_shock)
    N_train_nonshock = trainLabels.count(0)
    #print(N_train_nonshock)
    N_train = len(trainLabels)
    #print(N_train)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    print("Risk Accuracy for train data is:", Risk_Accuracy_train)
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    print("Risk Accuracy for test data is:", Risk_Accuracy_test)        
            
    # return precision, recall, f_measure, auc
    return train_predicted, test_predicted, test_predicted_prob_both, Risk_Accuracy_train, Risk_Accuracy_test


def learn_svm_MEASUREMENTS(trainData,trainLabels,testData,testLabels):
    print ("SVM Learning........")
    clf = svm.SVC(kernel='rbf', probability=True)
    param_grid = {"gamma" : [1, 1e-1, 1e-2, 1e-3], "C" : [1, 10, 100]}
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, n_jobs=5)
    clf.fit(trainData, trainLabels)
    train_predicted = clf.predict(trainData)
    train_accuracy = accuracy_score(trainLabels, train_predicted)
    
    test_predicted = clf.predict(testData)
    test_accuracy = accuracy_score(testLabels, test_predicted)
    
    classification_report_dict = classification_report(testLabels,test_predicted,digits=4,output_dict=True)
    positive_predictive_value = classification_report_dict['1.0']['precision']
    sensitivity = classification_report_dict['1.0']['recall']
    specificity = classification_report_dict['0.0']['recall']
        
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    #print(N_train_shock)
    N_train_nonshock = trainLabels.count(0)
    #print(N_train_nonshock)
    N_train = len(trainLabels)
    #print(N_train)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    auc = roc_auc_score(testLabels,test_predicted_prob)
    
    # return precision, recall, f_measure, auc
    return train_accuracy, test_accuracy, positive_predictive_value, sensitivity,\
        specificity, auc, Risk_Accuracy_train, Risk_Accuracy_test


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names)),key=lambda x: abs(x[0])))

    return [imp[::-1], names[::-1]]

    

def learn_svm_linear(trainData,trainLabels,testData,testLabels):
    trainData = pd.DataFrame(trainData, columns=[str(i) for i in range(trainData.shape[1])])
    print ("SVM Linear Learning........")
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(trainData, trainLabels)
    
    feature_names = [str(i) for i in range(trainData.shape[1])]
    feature_list = f_importances(clf.coef_[0], feature_names, top=10)

    train_predicted = clf.predict(trainData)
    train_accuracy = accuracy_score(trainLabels, train_predicted)
    
    test_predicted = clf.predict(testData)
    test_accuracy = accuracy_score(testLabels, test_predicted)
    
    classification_report_dict = classification_report(testLabels,test_predicted,digits=4,output_dict=True)
    print(classification_report(testLabels,test_predicted,digits=4))
    positive_predictive_value = classification_report_dict['1.0']['precision']
    sensitivity = classification_report_dict['1.0']['recall']
    specificity = classification_report_dict['0.0']['recall']
    
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    N_train_nonshock = trainLabels.count(0)
    N_train = len(trainLabels)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    print("Risk Accuracy for train data is:", Risk_Accuracy_train)
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    print("Risk Accuracy for test data is:", Risk_Accuracy_test)  
    
    auc = roc_auc_score(testLabels,test_predicted_prob)
    
    # return precision, recall, f_measure, auc
    return train_predicted, test_predicted, test_predicted_prob_both, feature_list, train_accuracy, test_accuracy, positive_predictive_value, sensitivity,\
        specificity, auc, Risk_Accuracy_train, Risk_Accuracy_test


def learn_svm_linear_MEASUREMENTS(trainData,trainLabels,testData,testLabels):
    trainData = pd.DataFrame(trainData, columns=[str(i) for i in range(trainData.shape[1])])
    print ("SVM Linear Learning........")
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(trainData, trainLabels)
    
    train_predicted = clf.predict(trainData)
    train_accuracy = accuracy_score(trainLabels, train_predicted)
    
    test_predicted = clf.predict(testData)
    test_accuracy = accuracy_score(testLabels, test_predicted)
    
    classification_report_dict = classification_report(testLabels,test_predicted,digits=4,output_dict=True)
    positive_predictive_value = classification_report_dict['1.0']['precision']
    sensitivity = classification_report_dict['1.0']['recall']
    specificity = classification_report_dict['0.0']['recall']
        
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    #print(N_train_shock)
    N_train_nonshock = trainLabels.count(0)
    #print(N_train_nonshock)
    N_train = len(trainLabels)
    #print(N_train)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    auc = roc_auc_score(testLabels,test_predicted_prob)
    
    # return precision, recall, f_measure, auc
    return train_accuracy, test_accuracy, positive_predictive_value, sensitivity,\
        specificity, auc, Risk_Accuracy_train, Risk_Accuracy_test




def learn_svm_linear_ALL(Data,Labels):
    Data = pd.DataFrame(Data, columns=[str(i) for i in range(Data.shape[1])])
    print ("SVM Learning........")
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(Data, Labels)
    feature_names = [str(i) for i in range(Data.shape[1])]
    feature_list = f_importances(clf.coef_[0], feature_names, top=10)
    predicted = clf.predict(Data)
    
    predicted_prob_both = clf.predict_proba(Data)
    #train data risk accuracy
    predicted_prob = predicted_prob_both[:,1]
    N_train_shock = Labels.count(1)
    #print(N_train_shock)
    N_train_nonshock = Labels.count(0)
    #print(N_train_nonshock)
    N_train = len(Labels)
    #print(N_train)
    total_value = 0
    for idx in range(N_train_shock):
        total_value += predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        total_value += 1-predicted_prob[idx]
    Risk_Accuracy = total_value/N_train
    print("Risk Accuracy is:", Risk_Accuracy)
    
    
    # return precision, recall, f_measure, auc
    return predicted, predicted_prob, feature_list, Risk_Accuracy




def learn_LR_ALL(Data,Labels):
    Data = pd.DataFrame(Data, columns=[str(i) for i in range(Data.shape[1])])
    print ("Logistic Regression Learning........")
    clf = LogisticRegression(solver = 'liblinear')
    clf.fit(Data, Labels)
    feature_names = [str(i) for i in range(Data.shape[1])]
    feature_list = f_importances(clf.coef_[0], feature_names, top=10)
    predicted = clf.predict(Data)
    
    predicted_prob_both = clf.predict_proba(Data)
    #train data risk accuracy
    predicted_prob = predicted_prob_both[:,1]
    N_train_shock = Labels.count(1)
    #print(N_train_shock)
    N_train_nonshock = Labels.count(0)
    #print(N_train_nonshock)
    N_train = len(Labels)
    #print(N_train)
    total_value = 0
    for idx in range(N_train_shock):
        total_value += predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        total_value += 1-predicted_prob[idx]
    Risk_Accuracy = total_value/N_train
    print("Risk Accuracy is:", Risk_Accuracy)
    
    
    # return precision, recall, f_measure, auc
    return predicted, predicted_prob, feature_list, Risk_Accuracy




def learn_LR(trainData,trainLabels,testData,testLabels):
    trainData = pd.DataFrame(trainData, columns=[str(i) for i in range(trainData.shape[1])])
    print ("Logistic Regression Learning........")
    clf = LogisticRegression(solver = 'liblinear')
    clf.fit(trainData, trainLabels)
    
    feature_names = [str(i) for i in range(trainData.shape[1])]
    feature_list = f_importances(clf.coef_[0], feature_names, top=10)
    
    train_predicted = clf.predict(trainData)
    train_accuracy = accuracy_score(trainLabels, train_predicted)
    
    test_predicted = clf.predict(testData)
    test_accuracy = accuracy_score(testLabels, test_predicted)
    
    classification_report_dict = classification_report(testLabels,test_predicted,digits=4,output_dict=True)
    print(classification_report(testLabels,test_predicted,digits=4))
    positive_predictive_value = classification_report_dict['1.0']['precision']
    sensitivity = classification_report_dict['1.0']['recall']
    specificity = classification_report_dict['0.0']['recall']
    
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    N_train_nonshock = trainLabels.count(0)
    N_train = len(trainLabels)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    print("Risk Accuracy for train data is:", Risk_Accuracy_train)
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    print("Risk Accuracy for test data is:", Risk_Accuracy_test)  
    
    auc = roc_auc_score(testLabels,test_predicted_prob)
    
    # return precision, recall, f_measure, auc
    return train_predicted, test_predicted, test_predicted_prob_both, feature_list, train_accuracy, test_accuracy, positive_predictive_value, sensitivity,\
        specificity, auc, Risk_Accuracy_train, Risk_Accuracy_test



def learn_LR_MEASUREMENTS(trainData,trainLabels,testData,testLabels):
    trainData = pd.DataFrame(trainData, columns=[str(i) for i in range(trainData.shape[1])])
    print ("Logistic Regression Learning........")
    clf = LogisticRegression(solver = 'liblinear')
    clf.fit(trainData, trainLabels)
    train_predicted = clf.predict(trainData)
    train_accuracy = accuracy_score(trainLabels, train_predicted)
    
    test_predicted = clf.predict(testData)
    test_accuracy = accuracy_score(testLabels, test_predicted)
    
    classification_report_dict = classification_report(testLabels,test_predicted,digits=4,output_dict=True)
    positive_predictive_value = classification_report_dict['1.0']['precision']
    sensitivity = classification_report_dict['1.0']['recall']
    specificity = classification_report_dict['0.0']['recall']
        
    train_predicted_prob_both = clf.predict_proba(trainData)
    test_predicted_prob_both = clf.predict_proba(testData)
    #train data risk accuracy
    train_predicted_prob = train_predicted_prob_both[:,1]
    N_train_shock = trainLabels.count(1)
    #print(N_train_shock)
    N_train_nonshock = trainLabels.count(0)
    #print(N_train_nonshock)
    N_train = len(trainLabels)
    #print(N_train)
    train_total_value = 0
    for idx in range(N_train_shock):
        train_total_value += train_predicted_prob[idx]
    for idx in range(N_train_shock, N_train):
        train_total_value += 1-train_predicted_prob[idx]
    Risk_Accuracy_train = train_total_value/N_train
    
    #test data risk accuracy
    test_predicted_prob = test_predicted_prob_both[:,1]
    N_test_shock = testLabels.count(1)
    N_test_nonshock = testLabels.count(0)
    N_test = len(testLabels)
    test_total_value = 0
    for idx in range(N_test_shock):
        test_total_value += test_predicted_prob[idx]
    for idx in range(N_test_shock, N_test):
        test_total_value += 1-test_predicted_prob[idx]
    Risk_Accuracy_test = test_total_value/N_test
    auc = roc_auc_score(testLabels,test_predicted_prob)
    
    # return precision, recall, f_measure, auc
    return train_accuracy, test_accuracy, positive_predictive_value, sensitivity,\
        specificity, auc, Risk_Accuracy_train, Risk_Accuracy_test
