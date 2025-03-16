
# Testing

import pandas as pd
import numpy as np
import pickle
import sys
import random
import math
import operator
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, classification_report,roc_auc_score, roc_curve, auc
from functools import reduce
import os
import xlsxwriter

import TemporalAbstraction
import RTPmining
import classifier

# Load the data files
def load_event():
    f = open('nonshock_event.pckl', 'rb')
    nonshock_event = pickle.load(f)
    f.close()
    f = open('shock_event.pckl', 'rb')
    shock_event = pickle.load(f)
    f.close()

    return shock_event, nonshock_event

# hour: prediction window length; obserr_h: obsevation window length
def make_MSS_new(dfLongCovid, dfNoLongCovid, pred_window, obs_window):
    print("ORIGINAL(Long Covid):", len(dfLongCovid.person_id.unique()))
    print("ORIGINAL(No Long Covid):", len(dfNoLongCovid.person_id.unique()))
      
    dfLongCovid = dfLongCovid[dfLongCovid.event_time - dfLongCovid.days_from_first_event >= pred_window]
    dfNoLongCovid = dfNoLongCovid[dfNoLongCovid.event_time - dfNoLongCovid.days_from_first_event >= pred_window]
    print("After excluding"+str(pred_window)+"days prediction window (Long-Covid):", len(dfLongCovid.person_id.unique()))
    print("After excluding"+str(pred_window)+"days prediction window (Long-Covid):", len(dfNoLongCovid.person_id.unique()))
    
    MSS_developed = []
    grouped = dfLongCovid.groupby('person_id')
    for name, group in grouped:
        group = group.sort_values(['days_from_first_event'])
        MSS_developed.append(TemporalAbstraction.MultivariateStateSequence(group))
        
    MSS_developed_final=[]
    #shockMSS_observationwindow_list = []
    item = MSS_developed[0]
    for item in MSS_developed:
        if len(item) != 0:
              end_time_list = [i.end for i in item]
              observation_window_length = max(end_time_list)-item[0].start
              #shockMSS_observationwindow_list.append(observation_window_length)
              if observation_window_length>=obs_window:
                MSS_developed_final.append(item)
    print("Numbers of satisfied MSS(Long Covid):", len(MSS_developed_final))
    
    MSS_nonLC = []
    grouped = dfNoLongCovid.groupby('person_id')
    for name, group in grouped:
        group = group.sort_values(['days_from_first_event'])
        MSS_nonLC.append(TemporalAbstraction.MultivariateStateSequence(group))

    MSS_nonLC_final = []
    #nonshockMSS_observationwindow_list = []
    for item in MSS_nonLC:
        if len(item) != 0:
              end_time_list = [i.end for i in item]
              #nonshockMSS_observationwindow_list.append(max(end_time_list)-item[0].start)
              if max(end_time_list)-item[0].start>=obs_window:
                MSS_nonLC_final.append(item)
    print("Numbers of satisfied MSS(No Long Covid):", len(MSS_nonLC_final))
    
    f = open('MSS_LC_'+'Pred'+str(pred_window)+'_Obser'+str(obs_window)+'.pckl', 'wb')
    pickle.dump(MSS_developed_final, f)
    f.close()
    f = open('MSS_nonLC_'+'Pred'+str(pred_window)+'_Obser'+str(obs_window)+'.pckl', 'wb')
    pickle.dump(MSS_nonLC_final, f)
    f.close()

    return MSS_developed_final, MSS_nonLC_final

################################# END OF NEW MAKE MSS (2020/04/13)

# hour: prediction window length; obserr_h: obsevation window length    
def load_MSS(pred_window, obs_window):
    f = open('MSS_LC_'+'Pred'+str(pred_window)+'_Obser'+str(obs_window)+'.pckl', 'rb')
    MSS_developed_final = pickle.load(f)
    f.close()
    f = open('MSS_nonLC_'+'Pred'+str(pred_window)+'_Obser'+str(obs_window)+'.pckl', 'rb')
    MSS_nonshock_final = pickle.load(f)
    f.close()

    return MSS_developed_final, MSS_nonshock_final


def store_patterns(i,trainC1_developed,trainC0, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues):
    C1_developed_patterns, kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK = \
        RTPmining.pattern_mining(trainC1_developed, g, g_int, sup_developed*len(trainC1_developed),featureValues)
    print("Total # patterns from developed:", len(C1_developed_patterns))
    C0_patterns, kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK = \
        RTPmining.pattern_mining(trainC0, g, g_int, sup_nonshock*len(trainC0),featureValues)
    print("Total # patterns from nonshock:", len(C0_patterns))
    
    ############## Writing patterns to the files #################
    C1_developed_patterns_file = open('C1_developed_Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.txt','w')
    C0Patterns_file = open('C0Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.txt','w')
    for p in C1_developed_patterns:
        C1_developed_patterns_file.write(p.describe())
    for p in C0_patterns:
        C0Patterns_file.write(p.describe())

    '''
    f1_developed__name = 'C1_developed_Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f0_name = 'C0Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f = open(f1_developed__name, 'wb')
    pickle.dump(C1_developed_patterns, f)
    f.close()
    f = open(f0_name, 'wb')
    pickle.dump(C0_patterns, f)
    f.close()
    '''

    return C1_developed_patterns, C0_patterns, \
        kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK, \
        kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK

def load_patterns(i, hour, obser_h, g, g_int, sup_developed, sup_nonshock):
    f1_developed__name = 'C1_developed_Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f0_name = 'C0Patterns_fold'+str(i)+'_Pred'+str(hour)+'_Obser'+str(obser_h)\
                                      +'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f = open(f1_developed__name, 'rb')
    C1_developed_patterns = pickle.load(f)
    f.close()
    f = open(f0_name, 'rb')
    C0_patterns = pickle.load(f)
    f.close()

    return C1_developed_patterns, C0_patterns



#allShockMSS = MSS_developed
#allNonshockMSS = MSS_nonshock    
def store_patterns_ALLDATA(allShockMSS, allNonshockMSS, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues):
    allShockMSS_patterns, kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK = \
        RTPmining.pattern_mining(allShockMSS, g, g_int, sup_developed*len(allShockMSS),featureValues)
    #print("Total # patterns from developed:", len(allShockMSS_patterns))
    
    allNonshockMSS_patterns, kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK = \
        RTPmining.pattern_mining(allNonshockMSS, g, g_int, sup_nonshock*len(allNonshockMSS),featureValues)
    #print("Total # patterns from nonshock:", len(allNonshockMSS_patterns))
    
    ############## Writing patterns to the files #################
    allShockMSS_patterns_file = open('allShockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.txt','w')
    allNonshockMSS_patterns_file = open('allNonshockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_nonshock)+'.txt','w')
    for p in allShockMSS_patterns:
        allShockMSS_patterns_file.write(p.describe())
    for p in allNonshockMSS_patterns:
        allNonshockMSS_patterns_file.write(p.describe())
    '''
    f1_shock_name = 'allShockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f0_nonshock_name = 'allNonshockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_nonshock)+'.pckl'
    f = open(f1_shock_name, 'wb')
    pickle.dump(allShockMSS_patterns, f)
    f.close()
    f = open(f0_nonshock_name, 'wb')
    pickle.dump(allNonshockMSS_patterns, f)
    f.close()
    ''' 
    return allShockMSS_patterns, allNonshockMSS_patterns, \
        kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK, \
        kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK

def load_patterns_ALLDATA(hour, obser_h, g, g_int, sup_developed, sup_nonshock):
    f1_shock_name = 'allShockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'.pckl'
    f0_nonshock_name = 'allNonshockMSS_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_nonshock)+'.pckl'
    f = open(f1_shock_name, 'rb')
    allShockMSS_patterns = pickle.load(f)
    f.close()
    f = open(f0_nonshock_name, 'rb')
    allNonshockMSS_patterns = pickle.load(f)
    f.close()

    return allShockMSS_patterns, allNonshockMSS_patterns

def random_subset( iterator, K ):
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item
    return result


def get_not_appear_pattern(patternlist1, patternlist2):
    new = []
    for p in patternlist1:
        if p not in patternlist2:
            new.append(p)
    
    return new


#pred_window = 0 # Prediction window in days
#obs_window = 0 # Observation window in days
#hour = pred_window
#obser_h = obs_window
# g:last interval length; g_int: inter state interval length; 
#ALL_shock_txt_name = 'shock_pattern_list_ALL.txt'
#ALL_nonshock_txt_name = 'nonshock_pattern_list_ALL.txt'
def noncontrast_HMM_ALLDATA(MSS_developed, MSS_nonshock, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues):  
    print("size of shock MSS:", len(MSS_developed))
    print("size of nonshock MSS:", len(MSS_nonshock))
    
    #########################################################
    # Objective 1: Find Important Patterns based on all data#
    #########################################################
    print("#############Task1_AllData#############")
    # ---- either store patterns or load the dumped ones
    shock_patterns, nonshock_patterns, kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK, \
        kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK \
            = store_patterns_ALLDATA(MSS_developed, MSS_nonshock, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues)
        
    allPatterns_shock = list(shock_patterns)
    print ("number of original shock patterns:", len(allPatterns_shock))
    allPatterns_nonshock = list(nonshock_patterns)
    print ("number of original nonshock patterns:", len(allPatterns_nonshock))
    # allPatterns_nonshock_NEW = []
    
    # for item in allPatterns_nonshock:
    #     if item not in allPatterns_shock:
    #         allPatterns_nonshock_NEW.append(item)
            
    allPatterns_nonshock_NEW = []
    MSS_idx_list_final_4D_NONSHOCK_flatten = [item for sublist in MSS_idx_list_final_4D_NONSHOCK for item in sublist]
    MSS_idx_SHOCK_NEW = [item for sublist in MSS_idx_list_final_4D_SHOCK for item in sublist]    
    MSS_idx_NONSHOCK_NEW = []
    
    for idxx in range(len(allPatterns_nonshock)):
        if allPatterns_nonshock[idxx] not in allPatterns_shock:
            allPatterns_nonshock_NEW.append(allPatterns_nonshock[idxx])
            MSS_idx_NONSHOCK_NEW.append(MSS_idx_list_final_4D_NONSHOCK_flatten[idxx])
    
    selected_pattern_shock = allPatterns_shock
    selected_pattern_nonshock = allPatterns_nonshock_NEW
             
    # final_endidx_return_ShockMSS_Contrast, final_MSSidx_return_ShockMSS_Contrast = \
    #         RTPmining.pattern_mss_opposite_match(MSS_developed, selected_pattern_shock, g, g_int)
        
    # print(1)
            
    # final_endidx_return_NonShockMSS_Contrast, final_MSSidx_return_NonShockMSS_Contrast = \
    #         RTPmining.pattern_mss_opposite_match(MSS_nonshock, selected_pattern_nonshock, g, g_int)
        
    # print(2)
        
    # length_list1 = [len(i) for i in MSS_idx_SHOCK_NEW]
    # length_list2 = [len(i) for i in MSS_idx_NONSHOCK_NEW]
        
    # All Patterns
    # idx_ary1 = np.argsort(length_list1)[:-(len(selected_pattern_shock)+1):-1]
    # idx_ary2 = np.argsort(length_list2)[:-(len(selected_pattern_nonshock)+1):-1]
    # top_pattern_shock = [selected_pattern_shock[i] for i in idx_ary1]    
    # top_pattern_nonshock = [selected_pattern_nonshock[i] for i in idx_ary2]  
    
    # idx_ary1 = np.argsort([len(i) for i in MSS_idx_SHOCK_NEW])[::-1]
    # idx_ary2 = np.argsort([len(i) for i in MSS_idx_NONSHOCK_NEW])[::-1]
    # Order the mined patterns first by shock patterns and non shock patterns then by the frequency in the shock group if it is a shock pattern
    # and frequency in the nonshock group if it is an nonshock pattern
    top_pattern_shock = [selected_pattern_shock[i] for i in np.argsort([len(i) for i in MSS_idx_SHOCK_NEW])[::-1]]    
    top_pattern_nonshock = [selected_pattern_nonshock[i] for i in np.argsort([len(i) for i in MSS_idx_NONSHOCK_NEW])[::-1]]  
    final_pattern_set = []
    final_pattern_set.extend(top_pattern_shock)
    final_pattern_set.extend(top_pattern_nonshock)
    print("total number of mined patterns:",len(final_pattern_set))
      
    # Obtain information for each mined patterns in all shock MSS
    final_endidx_return_ShockMSS_TRAIN, final_MSSidx_return_ShockMSS_TRAIN = \
            RTPmining.pattern_mss_opposite_match(MSS_developed, final_pattern_set, g, g_int)
    # Obtain information for each mined patterns in all nonshock MSS
    final_endidx_return_NonShockMSS_TRAIN, final_MSSidx_return_NonShockMSS_TRAIN = \
            RTPmining.pattern_mss_opposite_match(MSS_nonshock, final_pattern_set, g, g_int)
    
                 
    # Create binary matrix based on mined patterns (the second g in the int(g/g) is designed for l-RTP, for simple RTP, just use 1 (i.e., int(g/g)))
    trainBinaryMatrixSet_shock = np.zeros((len(MSS_developed), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
    trainBinaryMatrixSet_nonshock = np.zeros((len(MSS_nonshock), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
    for i in range(0, len(final_MSSidx_return_ShockMSS_TRAIN)):
        for j in range(0,len(final_MSSidx_return_ShockMSS_TRAIN[i])):
            end_idx_list = final_endidx_return_ShockMSS_TRAIN[i][j]
            instance_idx = final_MSSidx_return_ShockMSS_TRAIN[i][j]
            mss = MSS_developed[instance_idx]
            end_time_list = [item.end for item in mss]
            for item in end_idx_list:
                val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                if val == 1:
                    val = 0
                trainBinaryMatrixSet_shock[instance_idx][i][val] = 1   
    for i in range(0, len(final_MSSidx_return_NonShockMSS_TRAIN)):
        for j in range(0,len(final_MSSidx_return_NonShockMSS_TRAIN[i])):
            end_idx_list = final_endidx_return_NonShockMSS_TRAIN[i][j]
            instance_idx = final_MSSidx_return_NonShockMSS_TRAIN[i][j]
            mss = MSS_nonshock[instance_idx]
            end_time_list = [item.end for item in mss]
            for item in end_idx_list:
                val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                if val == 1:
                    val = 0
                trainBinaryMatrixSet_nonshock[instance_idx][i][val] = 1  
    binaryMatrix_trainshock = trainBinaryMatrixSet_shock.reshape(len(trainBinaryMatrixSet_shock), len(trainBinaryMatrixSet_shock[0]))
    binaryMatrix_trainnonshock = trainBinaryMatrixSet_nonshock.reshape(len(trainBinaryMatrixSet_nonshock), len(trainBinaryMatrixSet_nonshock[0]))    
    # binaryMatrix_trainshock = np.zeros((len(trainBinaryMatrixSet_shock),len(trainBinaryMatrixSet_shock[0])))
    # binaryMatrix_trainnonshock = np.zeros((len(trainBinaryMatrixSet_nonshock),len(trainBinaryMatrixSet_nonshock[0])))  
    # for i in range(0,len(trainBinaryMatrixSet_shock)):
    #     for j in range(0,len(trainBinaryMatrixSet_shock[i])):
    #         binaryMatrix_trainshock[i,j] = trainBinaryMatrixSet_shock[i][j]   
    # for i in range(0,len(trainBinaryMatrixSet_nonshock)):
    #     for j in range(0,len(trainBinaryMatrixSet_nonshock[i])):
    #         binaryMatrix_trainnonshock[i,j] = trainBinaryMatrixSet_nonshock[i][j]
    binaryMatrix_ALL = np.concatenate((binaryMatrix_trainshock,binaryMatrix_trainnonshock),axis=0)
    print("Binary matrix size:", binaryMatrix_ALL.shape)
        
    # Assign Labels
    Labels_ALL = list(np.ones(len(trainBinaryMatrixSet_shock)))
    Labels_ALL.extend(np.zeros(len(trainBinaryMatrixSet_nonshock)))
    print("Label size:", len(Labels_ALL))
    
    ###################################
    ## SVM Linear
    ###################################
    
    predicted, pred_prob, feature_list, Risk_Accuracy = classifier.learn_svm_linear_ALL(binaryMatrix_ALL,Labels_ALL)
        
    print("SVM Linear Kernel Results (ALL DATA):")
    train_accuracy_linear = accuracy_score(Labels_ALL, predicted)
    print(metrics.confusion_matrix(Labels_ALL,predicted))
    print(classification_report(Labels_ALL,predicted,digits=4))
    classification_report_dict = classification_report(Labels_ALL,predicted,digits=4,output_dict=True)
    auc_linear = roc_auc_score(Labels_ALL,pred_prob)
    print ("Area under ROC:", auc_linear)
    print ("Risk Accuracy:", Risk_Accuracy)
    
    final_result_tuple_list = []
    final_result_tuple_list.append(('training accuracy', train_accuracy_linear))
    final_result_tuple_list.append(('positive predictive value', classification_report_dict['1.0']['precision']))
    final_result_tuple_list.append(('sensitivity', classification_report_dict['1.0']['recall']))
    final_result_tuple_list.append(('specificity', classification_report_dict['0.0']['recall']))
    final_result_tuple_list.append(('auc', auc_linear))
    final_result_tuple_list.append(('RA', Risk_Accuracy))
    
    # Write the Measurement Results
    current_directory = os.getcwd() + '/All_Results'
    new_folder_name = r'Results_NonContrasted_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)
    final_directory = os.path.join(current_directory, new_folder_name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    excel_file_name_NEW = 'ALLDATA_SVMLinear_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'All_Results/'+ new_folder_name+'/'+excel_file_name_NEW)   
    worksheet = workbook.add_worksheet('Results')
    for row, line in enumerate(final_result_tuple_list):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()   
        
    # Write the Important Patterns
    current_dir = os.getcwd()
    excel_file_name = 'ALL_SVMLinear_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'ALL_SVM_Linear/'+excel_file_name)
       
    all_pattern_list = []
    for item in selected_pattern_shock:
        all_pattern_list.append(item.describe())
    for item in selected_pattern_nonshock:
        all_pattern_list.append(item.describe())
    feature_idx_list = [int(item) for item in feature_list[1]]
    pattern_list = []
    for item in feature_idx_list:
        pattern_list.append(all_pattern_list[item])
    feature_data = list(zip(pattern_list,feature_list[0]))
    worksheet = workbook.add_worksheet('Important_Patterns')
    for row, line in enumerate(feature_data):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()
    
    
    ###################################
    ## Logistic Regression
    ###################################
    predicted, pred_prob, feature_list, Risk_Accuracy = classifier.learn_LR_ALL(binaryMatrix_ALL,Labels_ALL)
        
    print("Logistic Regression Results (ALL DATA):")
    train_accuracy_LR = accuracy_score(Labels_ALL, predicted)
    print(metrics.confusion_matrix(Labels_ALL,predicted))
    print(classification_report(Labels_ALL,predicted,digits=4))
    classification_report_dict = classification_report(Labels_ALL,predicted,digits=4,output_dict=True)
    auc_LR = roc_auc_score(Labels_ALL,pred_prob)
    print ("Area under ROC:", auc_LR)
    print ("Risk Accuracy:", Risk_Accuracy)
    
    final_result_tuple_list = []
    final_result_tuple_list.append(('training accuracy', train_accuracy_LR))
    final_result_tuple_list.append(('positive predictive value', classification_report_dict['1.0']['precision']))
    final_result_tuple_list.append(('sensitivity', classification_report_dict['1.0']['recall']))
    final_result_tuple_list.append(('specificity', classification_report_dict['0.0']['recall']))
    final_result_tuple_list.append(('auc', auc_LR))
    final_result_tuple_list.append(('RA', Risk_Accuracy))
    
    # Write the Measurement Results
    excel_file_name_NEW = 'ALLDATA_LR_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'All_Results/'+\
                               new_folder_name+'/'+excel_file_name_NEW)   
    worksheet = workbook.add_worksheet('Results')
    for row, line in enumerate(final_result_tuple_list):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()   
    
    
    # Write the Important Patterns
    excel_file_name = 'ALL_LR_Patterns_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'ALL_LR/'+excel_file_name)

        
    all_pattern_list = []
    for item in selected_pattern_shock:
        all_pattern_list.append(item.describe())
    for item in selected_pattern_nonshock:
        all_pattern_list.append(item.describe())
    feature_idx_list = [int(item) for item in feature_list[1]]
    pattern_list = []
    for item in feature_idx_list:
        pattern_list.append(all_pattern_list[item])
    feature_data = list(zip(pattern_list,feature_list[0]))
    worksheet = workbook.add_worksheet('Important_Patterns')
    for row, line in enumerate(feature_data):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()
    
    return trainBinaryMatrixSet_shock, trainBinaryMatrixSet_nonshock, top_pattern_shock, top_pattern_nonshock, binaryMatrix_ALL



#hour = pred_window
#obser_h = obs_window   
def noncontrast_HMM(MSS_developed, MSS_nonshock, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues): 
    
    print ("size of nonshock data:", len(MSS_nonshock))
    print ("size of shock data:", len(MSS_developed))
    
    #####################################################
    # Objective 2: Five-Fold Cross Validation For Prediction
    #####################################################
    print("#############Task2_Five-Fold CV Prediction#############")
    num_folds = 5
    C0_subset_size = len(MSS_nonshock)//num_folds
    C1_developed_subset_size = len(MSS_developed)//num_folds
    
    Linear_Five_TrainA, Linear_Five_TestA, Linear_Five_PPV, Linear_Five_Sen, Linear_Five_Spe, Linear_Five_AUC, Linear_Five_TrainRA, Linear_Five_TestRA = [],[],[],[],[],[],[],[]
    LR_Five_TrainA, LR_Five_TestA, LR_Five_PPV, LR_Five_Sen, LR_Five_Spe, LR_Five_AUC, LR_Five_TrainRA, LR_Five_TestRA = [],[],[],[],[],[],[],[]
    test_labels, train_labels = [], []
    test_pred_linear, test_pred_prob_linear, train_pred_linear, feature_importance_linear = [], [], [], []
    test_pred_LR, test_pred_prob_LR, train_pred_LR, feature_importance_LR = [], [], [], []
    train_shock_fold_list, train_nonshock_fold_list, test_shock_fold_list, test_nonshock_fold_list = [],[],[],[]
    shock_fold_list, nonshock_fold_list = [],[]
    binaryMatrix_train_all, binaryMatrix_test_all = [], []
    SVM_Linear_RA_train, SVM_Linear_RA_test, LR_RA_train, LR_RA_test = [], [], [], []
    #New_shock, New_nonshock = [],[]
    
    for i in range(num_folds):
        print( "***************** FOLD ", i+1, "*****************")
        trainC0 = MSS_nonshock[:i*C0_subset_size] + MSS_nonshock[(i+1)*C0_subset_size:]
        trainC1_developed = MSS_developed[:i*C1_developed_subset_size] + MSS_developed[(i+1)*C1_developed_subset_size:]
        testC0 = MSS_nonshock[i*C0_subset_size:][:C0_subset_size]
        testC1_developed = MSS_developed[i*C1_developed_subset_size:][:C1_developed_subset_size]
        print ("Size of shock and nonshock training:", len(trainC1_developed), len(trainC0))
        print( "Size of shock and nonshock test:", len(testC1_developed), len(testC0))
        
        # if settings == 'trunc':
        #     testC0 = MSS_nonshock[i*C0_subset_size:][:C0_subset_size]
        #     testC1_developed = MSS_developed[i*C1_developed_subset_size:][:C1_developed_subset_size]
        
        # ---- either store patterns or load the dumped ones
        C1_developed_patterns, C0_patterns, kRTP_idx_list_final_4D_SHOCK, MSS_idx_list_final_4D_SHOCK, \
        kRTP_idx_list_final_4D_NONSHOCK, MSS_idx_list_final_4D_NONSHOCK \
            = store_patterns(i,trainC1_developed, trainC0, hour, obser_h, g, g_int, sup_developed, sup_nonshock,featureValues)
    
        allPatterns_shock = list(C1_developed_patterns)
        print ("number of original shock patterns:", len(allPatterns_shock))
        allPatterns_nonshock = list(C0_patterns)
        print ("number of original nonshock patterns:", len(allPatterns_nonshock))
        allPatterns_nonshock_NEW = []
        MSS_idx_list_final_4D_NONSHOCK_flatten = [item for sublist in MSS_idx_list_final_4D_NONSHOCK for item in sublist]
        MSS_idx_SHOCK_NEW = [item for sublist in MSS_idx_list_final_4D_SHOCK for item in sublist]    
        MSS_idx_NONSHOCK_NEW = []
        
        for idxx in range(len(allPatterns_nonshock)):
            if allPatterns_nonshock[idxx] not in allPatterns_shock:
                allPatterns_nonshock_NEW.append(allPatterns_nonshock[idxx])
                MSS_idx_NONSHOCK_NEW.append(MSS_idx_list_final_4D_NONSHOCK_flatten[idxx])
        
        selected_pattern_shock = allPatterns_shock
        selected_pattern_nonshock = allPatterns_nonshock_NEW
        
        # Order the mined patterns first by shock patterns and non shock patterns then by the frequency in the shock group if it is a shock pattern
        # and frequency in the nonshock group if it is an nonshock pattern
        top_pattern_shock = [selected_pattern_shock[i] for i in np.argsort([len(i) for i in MSS_idx_SHOCK_NEW])[::-1]]    
        top_pattern_nonshock = [selected_pattern_nonshock[i] for i in np.argsort([len(i) for i in MSS_idx_NONSHOCK_NEW])[::-1]]  
        final_pattern_set = []
        final_pattern_set.extend(top_pattern_shock)
        final_pattern_set.extend(top_pattern_nonshock)
        print("total number of mined patterns:",len(final_pattern_set))
          
        
        # Obtain information for each mined patterns in all train shock MSS
        final_endidx_return_ShockMSS_TRAIN, final_MSSidx_return_ShockMSS_TRAIN = \
            RTPmining.pattern_mss_opposite_match(trainC1_developed, final_pattern_set, g, g_int)
        # Obtain information for each mined patterns in all train nonshock MSS
        final_endidx_return_NonShockMSS_TRAIN, final_MSSidx_return_NonShockMSS_TRAIN = \
            RTPmining.pattern_mss_opposite_match(trainC0, final_pattern_set, g, g_int)
        # Obtain information for each mined patterns in all test shock MSS  
        final_endidx_return_ShockMSS_TEST, final_MSSidx_return_ShockMSS_TEST = \
            RTPmining.pattern_mss_opposite_match(testC1_developed, final_pattern_set, g, g_int)
        # Obtain information for each mined patterns in all test nonshock MSS   
        final_endidx_return_NonShockMSS_TEST, final_MSSidx_return_NonShockMSS_TEST = \
            RTPmining.pattern_mss_opposite_match(testC0, final_pattern_set, g, g_int)

        
        # Create binary matrix based on mined patterns (the second g in the int(g/g) is designed for l-RTP, for simple RTP, just use 1 (i.e., int(g/g)))
        trainBinaryMatrixSet_shock = np.zeros((len(trainC1_developed), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
        trainBinaryMatrixSet_nonshock = np.zeros((len(trainC0), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
        testBinaryMatrixSet_shock = np.zeros((len(testC1_developed), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
        testBinaryMatrixSet_nonshock = np.zeros((len(testC0), len(selected_pattern_shock)+len(selected_pattern_nonshock), int(g/g)))
        # Train Shock
        for i in range(0, len(final_MSSidx_return_ShockMSS_TRAIN)):
            for j in range(0,len(final_MSSidx_return_ShockMSS_TRAIN[i])):
                end_idx_list = final_endidx_return_ShockMSS_TRAIN[i][j]
                instance_idx = final_MSSidx_return_ShockMSS_TRAIN[i][j]
                mss = trainC1_developed[instance_idx]
                end_time_list = [item.end for item in mss]
                for item in end_idx_list:
                    val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                    if val == 1:
                        val = 0
                    trainBinaryMatrixSet_shock[instance_idx][i][val] = 1         
        # Train Nonshock
        for i in range(0, len(final_MSSidx_return_NonShockMSS_TRAIN)):
            for j in range(0,len(final_MSSidx_return_NonShockMSS_TRAIN[i])):
                end_idx_list = final_endidx_return_NonShockMSS_TRAIN[i][j]
                instance_idx = final_MSSidx_return_NonShockMSS_TRAIN[i][j]
                mss = trainC0[instance_idx]
                end_time_list = [item.end for item in mss]
                for item in end_idx_list:
                    val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                    if val == 1:
                        val = 0
                    trainBinaryMatrixSet_nonshock[instance_idx][i][val] = 1  
        # Test Shock
        for i in range(0, len(final_MSSidx_return_ShockMSS_TEST)):
            for j in range(0,len(final_MSSidx_return_ShockMSS_TEST[i])):
                end_idx_list = final_endidx_return_ShockMSS_TEST[i][j]
                instance_idx = final_MSSidx_return_ShockMSS_TEST[i][j]
                mss = testC1_developed[instance_idx]
                end_time_list = [item.end for item in mss]
                for item in end_idx_list:
                    val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                    if val == 1:
                        val = 0
                    testBinaryMatrixSet_shock[instance_idx][i][val] = 1            
        # Test Nonshock
        for i in range(0, len(final_MSSidx_return_NonShockMSS_TEST)):
            for j in range(0,len(final_MSSidx_return_NonShockMSS_TEST[i])):
                end_idx_list = final_endidx_return_NonShockMSS_TEST[i][j]
                instance_idx = final_MSSidx_return_NonShockMSS_TEST[i][j]
                mss = testC0[instance_idx]
                end_time_list = [item.end for item in mss]
                for item in end_idx_list:
                    val = int(g/g)-math.ceil((max(end_time_list)-mss[item[-1]].end)/g)
                    if val == 1:
                        val = 0
                    testBinaryMatrixSet_nonshock[instance_idx][i][val] = 1  
        binaryMatrix_trainshock = trainBinaryMatrixSet_shock.reshape(len(trainBinaryMatrixSet_shock), len(trainBinaryMatrixSet_shock[0]))
        binaryMatrix_trainnonshock = trainBinaryMatrixSet_nonshock.reshape(len(trainBinaryMatrixSet_nonshock), len(trainBinaryMatrixSet_nonshock[0]))   
        binaryMatrix_testshock = testBinaryMatrixSet_shock.reshape(len(testBinaryMatrixSet_shock), len(testBinaryMatrixSet_shock[0]))
        binaryMatrix_testnonshock =testBinaryMatrixSet_nonshock.reshape(len(testBinaryMatrixSet_nonshock), len(testBinaryMatrixSet_nonshock[0]))   
        binaryMatrix_train = np.concatenate((binaryMatrix_trainshock,binaryMatrix_trainnonshock),axis=0)
        binaryMatrix_test = np.concatenate((binaryMatrix_testshock,binaryMatrix_testnonshock),axis=0)
        print("Binary matrix size (Train):", binaryMatrix_train.shape)
        print("Binary matrix size (Test):", binaryMatrix_test.shape)
        
            
        # Assign Labels
        trainLabels = list(np.ones(len(trainBinaryMatrixSet_shock)))
        trainLabels.extend(np.zeros(len(trainBinaryMatrixSet_nonshock)))
        print("Label size (Train):", len(trainLabels))
        testLabels = list(np.ones(len(testBinaryMatrixSet_shock)))
        testLabels.extend(np.zeros(len(testBinaryMatrixSet_nonshock)))
        print("Label size (Test):", len(testLabels))
        
        train_labels.extend(trainLabels)
        test_labels.extend(testLabels)
        
        # Learn the linear SVM model
        trp_linear, tsp_linear, tsp_prob_linear, feature_list, TrainA_linear, TestA_linear, PPV_linear, Sen_linear, Spe_linear, AUC_linear, TrainRA_linear, TestRA_linear\
            = classifier.learn_svm_linear(binaryMatrix_train,trainLabels,binaryMatrix_test,testLabels)
        train_pred_linear.extend(trp_linear)
        test_pred_linear.extend(tsp_linear)
        test_pred_prob_linear.extend([[el[1] for el in tsp_prob_linear]])
        feature_importance_linear.append(feature_list)
        SVM_Linear_RA_train.append(TrainRA_linear)
        SVM_Linear_RA_test.append(TestRA_linear)
        
        Linear_Five_TrainA.append(TrainA_linear)
        Linear_Five_TestA.append(TestA_linear)
        Linear_Five_PPV.append(PPV_linear)
        Linear_Five_Sen.append(Sen_linear)
        Linear_Five_Spe.append(Spe_linear)
        Linear_Five_AUC.append(AUC_linear)
        Linear_Five_TrainRA.append(TrainRA_linear)
        Linear_Five_TestRA.append(TestRA_linear)
        
        # Learn the Logistic Regression model
        trp_LR, tsp_LR, tsp_prob_LR, feature_list_LR, TrainA_LR, TestA_LR, PPV_LR, Sen_LR, Spe_LR, AUC_LR, TrainRA_LR, TestRA_LR\
            = classifier.learn_LR(binaryMatrix_train,trainLabels,binaryMatrix_test,testLabels)
        train_pred_LR.extend(trp_LR)
        test_pred_LR.extend(tsp_LR)
        test_pred_prob_LR.extend([[el[1] for el in tsp_prob_LR]])
        feature_importance_LR.append(feature_list_LR)
        LR_RA_train.append(TrainRA_LR)
        LR_RA_test.append(TestRA_LR)
        
        LR_Five_TrainA.append(TrainA_LR)
        LR_Five_TestA.append(TestA_LR)
        LR_Five_PPV.append(PPV_LR)
        LR_Five_Sen.append(Sen_LR)
        LR_Five_Spe.append(Spe_LR)
        LR_Five_AUC.append(AUC_LR)
        LR_Five_TrainRA.append(TrainRA_LR)
        LR_Five_TestRA.append(TestRA_LR)
        
        # Store Other Data
        train_shock_fold_list.append(trainBinaryMatrixSet_shock) 
        train_nonshock_fold_list.append(trainBinaryMatrixSet_nonshock)
        test_shock_fold_list.append(testBinaryMatrixSet_shock)
        test_nonshock_fold_list.append(testBinaryMatrixSet_nonshock)
        shock_fold_list.append(top_pattern_shock)
        nonshock_fold_list.append(top_pattern_nonshock)
        binaryMatrix_train_all.append(binaryMatrix_train)
        binaryMatrix_test_all.append(binaryMatrix_test)
    
    
    current_directory = os.getcwd() + '/All_Results'
    new_folder_name = r'Results_NonContrasted_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'\
        +str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)
    final_directory = os.path.join(current_directory, new_folder_name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory) 
    
    # Write the Measurement Results (SVM Linear) (Five means 5 fold but we now use ten folds)
    df_Linear = pd.DataFrame({'training accuracy': Linear_Five_TrainA,
                           'test accuracy': Linear_Five_TestA,
                           'positive predictive value': Linear_Five_PPV,
                           'sensitivity': Linear_Five_Sen,
                           'specificity': Linear_Five_Spe,
                           'auc': Linear_Five_AUC,
                           'train RA': Linear_Five_TrainRA,
                           'test RA': Linear_Five_TestRA})
    excel_file_name_NEW = 'Ten_SVMLinear_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'\
        +str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.csv'
    df_Linear.to_csv(os.getcwd()+'/'+'All_Results/'+\
                               new_folder_name+'/'+excel_file_name_NEW)
    
    # Write the Average Measurement Results (SVM Linear)
    print("SVM Linear Kernel Results:")
    train_accuracy_linear = accuracy_score(train_labels, train_pred_linear)
    print ("training accuracy:", train_accuracy_linear)
    test_accuracy_linear = accuracy_score(test_labels, test_pred_linear)
    print ("test accuracy:", test_accuracy_linear)
    print(metrics.confusion_matrix(test_labels,test_pred_linear))
    print(classification_report(test_labels,test_pred_linear,digits=4))
    classification_report_dict = classification_report(test_labels,test_pred_linear,digits=4,output_dict=True)
    test_pred_prob_new_linear = reduce(lambda x,y: x+y,test_pred_prob_linear)
    auc_linear = roc_auc_score(test_labels,test_pred_prob_new_linear)
    print ("Area under ROC:", auc_linear)
    print ("CV Average Risk Accuracy (train):", np.mean(SVM_Linear_RA_train))
    print ("CV Average Risk Accuracy (test):", np.mean(SVM_Linear_RA_test))
    
    final_result_tuple_list = []
    final_result_tuple_list.append(('training accuracy', train_accuracy_linear))
    final_result_tuple_list.append(('test accuracy', test_accuracy_linear))
    final_result_tuple_list.append(('positive predictive value', classification_report_dict['1.0']['precision']))
    final_result_tuple_list.append(('sensitivity', classification_report_dict['1.0']['recall']))
    final_result_tuple_list.append(('specificity', classification_report_dict['0.0']['recall']))
    final_result_tuple_list.append(('auc', auc_linear))
    final_result_tuple_list.append(('train average RA', np.mean(SVM_Linear_RA_train)))
    final_result_tuple_list.append(('test average RA', np.mean(SVM_Linear_RA_test)))
    
    excel_file_name_NEW = 'SVM_Linear_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'\
        +str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'All_Results/'+\
                               new_folder_name+'/'+excel_file_name_NEW)   
    worksheet = workbook.add_worksheet('Results')
    for row, line in enumerate(final_result_tuple_list):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()
    
    
    
    # Write the Measurement Results (LR)
    df_LR = pd.DataFrame({'training accuracy': LR_Five_TrainA,
                           'test accuracy': LR_Five_TestA,
                           'positive predictive value': LR_Five_PPV,
                           'sensitivity': LR_Five_Sen,
                           'specificity': LR_Five_Spe,
                           'auc': LR_Five_AUC,
                           'train RA': LR_Five_TrainRA,
                           'test RA': LR_Five_TestRA})
    excel_file_name_NEW = 'Ten_LR_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'\
        +str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.csv'
    df_LR.to_csv(os.getcwd()+'/'+'All_Results/'+\
                               new_folder_name+'/'+excel_file_name_NEW)
    # Write the Average Measurement Results (LR)
    print("Logistic Regression Results:")
    train_accuracy_LR = accuracy_score(train_labels, train_pred_LR)
    print ("training accuracy:", train_accuracy_LR)
    test_accuracy_LR = accuracy_score(test_labels, test_pred_LR)
    print ("test accuracy:", test_accuracy_LR)
    print(metrics.confusion_matrix(test_labels,test_pred_LR))
    print(classification_report(test_labels,test_pred_LR,digits=4))
    classification_report_dict = classification_report(test_labels,test_pred_LR,digits=4,output_dict=True)
    test_pred_prob_new_LR = reduce(lambda x,y: x+y,test_pred_prob_LR)
    auc_LR = roc_auc_score(test_labels,test_pred_prob_new_LR)
    print ("Area under ROC:", auc_LR)
    print ("CV Average Risk Accuracy (train):", np.mean(LR_RA_train))
    print ("CV Average Risk Accuracy (test):", np.mean(LR_RA_test))
    
    final_result_tuple_list = []
    final_result_tuple_list.append(('training accuracy', train_accuracy_LR))
    final_result_tuple_list.append(('test accuracy', test_accuracy_LR))
    final_result_tuple_list.append(('positive predictive value', classification_report_dict['1.0']['precision']))
    final_result_tuple_list.append(('sensitivity', classification_report_dict['1.0']['recall']))
    final_result_tuple_list.append(('specificity', classification_report_dict['0.0']['recall']))
    final_result_tuple_list.append(('auc', auc_LR))
    final_result_tuple_list.append(('train average RA', np.mean(LR_RA_train)))
    final_result_tuple_list.append(('test average RA', np.mean(LR_RA_test)))
    
    excel_file_name_NEW = 'LR_Pred'+str(hour)+'_Obser'+str(obser_h)+'_G'\
        +str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'All_Results/'+\
                               new_folder_name+'/'+excel_file_name_NEW)   
    worksheet = workbook.add_worksheet('Results')
    for row, line in enumerate(final_result_tuple_list):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
    workbook.close()
    

    return train_shock_fold_list, train_nonshock_fold_list, test_shock_fold_list, test_nonshock_fold_list, \
         shock_fold_list, nonshock_fold_list, binaryMatrix_train_all, binaryMatrix_test_all, \
             feature_importance_linear, feature_importance_LR