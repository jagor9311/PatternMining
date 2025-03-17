import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import xlsxwriter
import math

import os # Package to change your working directory
# Change Directory
#os.chdir('C:/Users/agorj/Box/Agor Research/NIH Long Covid Computational Challenge/Contrasted Pattern Mining New')
os.chdir(r'C:\Users\Joe\PycharmProjects\PatternMining')
os.getcwd()

import TemporalAbstraction
import RTPmining
import classifier
import main


# Global Variables
Features = ['condition_concept_name','drug_concept_name','device_concept_name']


###########################################################
######## START:  Read and Pre-Process in Data #############
###########################################################


##################
## Read in Data ##
##################
if True:
    '''
    condition_occurrence = pd.read_excel('C:/Users/Joe/Box/Agor Research/NIH Long Covid Computational Challenge/Contrasted Pattern Mining New/Sample_Pattern_Mining_Data.xlsx',sheet_name="condition_occurrence")
    drug_exposure = pd.read_excel('C:/Users/Joe/Box/Agor Research/NIH Long Covid Computational Challenge/Contrasted Pattern Mining New/Sample_Pattern_Mining_Data.xlsx',sheet_name="drug_exposure")
    device_exposure = pd.read_excel('C:/Users/Joe/Box/Agor Research/NIH Long Covid Computational Challenge/Contrasted Pattern Mining New/Sample_Pattern_Mining_Data.xlsx',sheet_name="device_exposure")
    long_COVID_Silver_Standard = pd.read_excel('C:/Users/Joe/Box/Agor Research/NIH Long Covid Computational Challenge/Contrasted Pattern Mining New/Sample_Pattern_Mining_Data.xlsx',sheet_name="long_COVID_Silver_Standard")
    '''
    condition_occurrence = pd.read_excel(r'C:\Users\Joe\PycharmProjects\PatternMining\Sample_Pattern_Mining_Data.xlsx',sheet_name="condition_occurrence")
    drug_exposure = pd.read_excel(r'C:\Users\Joe\PycharmProjects\PatternMining\Sample_Pattern_Mining_Data.xlsx',sheet_name="drug_exposure")
    device_exposure = pd.read_excel(r'C:\Users\Joe\PycharmProjects\PatternMining\Sample_Pattern_Mining_Data.xlsx',sheet_name="device_exposure")
    long_COVID_Silver_Standard = pd.read_excel(r'C:\Users\Joe\PycharmProjects\PatternMining\Sample_Pattern_Mining_Data.xlsx',sheet_name="long_COVID_Silver_Standard")
    
   
########################
## Data Preprocessing ##
########################

# Merge dataframes by person_id and date
df = pd.merge(condition_occurrence,device_exposure, how = 'outer', on = ['person_id','date'])
df = pd.merge(df,drug_exposure, how = 'outer', on = ['person_id','date'])

# Remove duplicate condition, device, druge, and date combinations
df1 = df.drop_duplicates(
    subset = ['person_id','condition_concept_name', 'device_concept_name','drug_concept_name','date'],
    keep = 'last').reset_index(drop = True)

# Sort values by date and person and reset the index
df2 = df1.sort_values(by = ['person_id', 'date'], ascending = [True, True], na_position = 'first')
df2 = df2.reset_index()

# Calculate unique persons
person_ids = pd.unique(df2['person_id'])
n = len(person_ids) 

# Calculate the days from first observation as for each individual and store in temporary list
temp = []
for i in person_ids:
    df3 = df2[df2['person_id']==i]
    df3 = df3.reset_index()
    temp1 = [(df3.loc[x,'date']-df3.loc[0,'date']).days for x in range(df3.shape[0])]
    temp = temp+temp1

# Insert a new column into data frame that provides the days from the first observation
df2.insert(2,'days_from_first_event',temp)
df2 = df2.drop('index',axis=1) # Remove a created index column


# Split data into case (long-covid) and control (no long-covid)
dfLongCovid = pd.DataFrame()
dfNoLongCovid = pd.DataFrame()
for i in person_ids:
    val = long_COVID_Silver_Standard.loc[long_COVID_Silver_Standard['person_id'] == i, 'time_to_pasc']
    #if math.isnan(long_COVID_Silver_Standard[long_COVID_Silver_Standard['person_id'] == i]['time_to_pasc']):
    if math.isnan(val.iloc[0]):
        df3 = df2[df2['person_id']==i]
        df3 = df3.reset_index()
        temp = [df3.loc[df3.shape[0]-1,'days_from_first_event']]*df3.shape[0]
        df3['event_time'] = temp
        dfNoLongCovid = pd.concat([dfNoLongCovid, df3], ignore_index=True)
    else:
        df3 = df2[df2['person_id']==i]
        df3 = df3.reset_index()
        temp_lc = long_COVID_Silver_Standard[long_COVID_Silver_Standard['person_id']==i]['time_to_pasc'].values[0]
        temp_lc = long_COVID_Silver_Standard[long_COVID_Silver_Standard['person_id']==i]['COVID Index']+pd.to_timedelta(temp_lc,unit='D')
        temp_lc = temp_lc.tolist()
        temp = temp_lc[0] - df3.loc[0,'date']
        temp = [temp.days]*df3.shape[0]
        df3['event_time'] = temp
        dfLongCovid = pd.concat([dfLongCovid, df3], ignore_index=True)
      
dfLongCovid = dfLongCovid.drop('index',axis=1) # Remove a created index column
dfNoLongCovid = dfNoLongCovid.drop('index',axis=1) # Remove a created index column 

# Identify unique values of features
featureValues = []
for i in range(len(Features)):
    temp = list(pd.unique(dfLongCovid[Features[i]]))
    temp = [x for x in temp if str(x) != 'nan']
    temp1 = list(pd.unique(dfNoLongCovid[Features[i]]))
    temp1 = [x for x in temp1 if str(x) != 'nan']
    temp2 = temp+temp1
    temp3 = [*set(temp2)]
    featureValues.append(temp3)

# Write out updated population (block group) information
dfLongCovid.to_csv('merged_data_LongCovid.csv',index = False)  
dfNoLongCovid.to_csv('merged_data_NoLongCovid.csv',index = False)   


# Calculate Five Number Summaries for Time Differences
temp = pd.unique(dfLongCovid['days_from_first_event'])
temp = [x - temp[i - 1] for i, x in enumerate(temp) if i > 0]
temp = [temp[i] for i in range(len(temp)) if temp[i] >= 0]

# calculate quartiles
quartiles = np.percentile(temp, [25, 50, 75])
# calculate min/max
data_min, data_max = min(temp), max(temp)
# print 5-number summary
print("Five Number Summary for Time Difference Between Events for Long Covid Patients")
print('  Min: %.3f' % data_min)
print('  Q1: %.3f' % quartiles[0])
print('  Median: %.3f' % quartiles[1])
print('  Mean: %.3f' % (sum(temp)/len(temp)))
print('  Q3: %.3f' % quartiles[2])
print('  Max: %.3f\n' % data_max)

temp_interval = sum(temp)/len(temp)


temp = pd.unique(dfNoLongCovid['days_from_first_event'])
temp = [x - temp[i - 1] for i, x in enumerate(temp) if i > 0]
temp = [temp[i] for i in range(len(temp)) if temp[i] >= 0]

# calculate quartiles
quartiles = np.percentile(temp, [25, 50, 75])
# calculate min/max
data_min, data_max = min(temp), max(temp)
# print 5-number summary
print("Five Number Summary for Time Difference Between Events for Non Long Covid Patients")
print('  Min: %.3f' % data_min)
print('  Q1: %.3f' % quartiles[0])
print('  Median: %.3f' % quartiles[1])
print('  Mean: %.3f' % (sum(temp)/len(temp)))
print('  Q3: %.3f' % quartiles[2])
print('  Max: %.3f' % data_max)

if sum(temp)/len(temp)< temp_interval: temp_interval = sum(temp)/len(temp)

###########################################
######## PartA. Convert EHR to MSS ########
###########################################

# Parameters
pred_window = 0 # Prediction window in days
obs_window = 0 # Observation window in days


# If MSS is not stored yet
MSS_developed, MSS_nonshock = main.make_MSS_new(dfLongCovid, dfNoLongCovid, pred_window, obs_window)
# If MSS is already stored
#MSS_developed, MSS_nonshock = main.load_MSS(pred_window, obs_window)

g = 60 # Last interval length in days
g_int = temp_interval # Inter-state interval length (days)
sup_developed = 0.3
sup_nonshock = 0.3
num_case = len(pd.unique(dfLongCovid['person_id']))
num_control = len(pd.unique(dfNoLongCovid['person_id']))


###########################################
######## PartB. Pattern Mining ############
###########################################
# Mining patterns on the full data set
trainBinaryMatrixSet_shock, trainBinaryMatrixSet_nonshock,\
        top_pattern_shock_ALL, top_pattern_nonshock_ALL, binaryMatrix_ALL\
            = main.noncontrast_HMM_ALLDATA(MSS_developed, MSS_nonshock, pred_window, obs_window, g, g_int, sup_developed, sup_nonshock,featureValues)
# Mining patterns and evaluate mined RTPs using ten fold cross validation
trainM_shock_fold, trainM_nonshock_fold, testM_shock_fold, testM_nonshock_fold, \
    shock_list_fold, nonshock_list_fold, binaryMatrix_train_all, binaryMatrix_test_all,\
        feature_importance_linear, feature_importance_LR\
        = main.noncontrast_HMM(MSS_developed, MSS_nonshock, pred_window, obs_window, g, g_int, sup_developed, sup_nonshock,featureValues)

###################################################################################################
######## PartC. Save mined patterns and their model coefficients for each fold (not necessary)#####
###################################################################################################

# ########################################################################
### Write SVM Linear Important Features to Files #########
# ########################################################################
current_dir = os.getcwd()
excel_file_name = 'SVMLinear_Patterns_Pred'+str(pred_window)+'_Obser'+str(obs_window)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'SVM_Linear/'+excel_file_name)
for fold_num in range(5):
    top_pattern_shock = shock_list_fold[fold_num]
    top_pattern_nonshock = nonshock_list_fold[fold_num]
    feature_list = feature_importance_linear[fold_num]
    all_pattern_list = []
    for item in top_pattern_shock:
        all_pattern_list.append(item.describe())
    for item in top_pattern_nonshock:
        all_pattern_list.append(item.describe())
    feature_idx_list = [int(item) for item in feature_list[1]]
    pattern_list = []
    for item in feature_idx_list:
        pattern_list.append(all_pattern_list[item])
    feature_data = list(zip(pattern_list,feature_list[0]))
    worksheet = workbook.add_worksheet('Fold'+str(fold_num))
    for row, line in enumerate(feature_data):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
workbook.close()

# ########################################################################
### Write Logistic Regression Important Features to Files #########
# ########################################################################
current_dir = os.getcwd()
excel_file_name = 'LR_Patterns_Pred'+str(pred_window)+'_Obser'+str(obs_window)+'_G'+str(g)+'_g'+str(g_int)+'_sigma'+str(sup_developed)+'_'+str(sup_nonshock)+'.xlsx'
workbook = xlsxwriter.Workbook(os.getcwd()+'/'+'LR/'+excel_file_name)
for fold_num in range(5):
    top_pattern_shock = shock_list_fold[fold_num]
    top_pattern_nonshock = nonshock_list_fold[fold_num]
    feature_list = feature_importance_LR[fold_num]
    all_pattern_list = []
    for item in top_pattern_shock:
        all_pattern_list.append(item.describe())
    for item in top_pattern_nonshock:
        all_pattern_list.append(item.describe())
    feature_idx_list = [int(item) for item in feature_list[1]]
    pattern_list = []
    for item in feature_idx_list:
        pattern_list.append(all_pattern_list[item])
    feature_data = list(zip(pattern_list,feature_list[0]))
    worksheet = workbook.add_worksheet('Fold'+str(fold_num))
    for row, line in enumerate(feature_data):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
workbook.close()

# ###############################################################################
# # PartD. Preprocessing Before Extracting MPTP and maixmal RTPs (Ten Fold) #####
# ###############################################################################
for idxx in range(5):
    trainM_shock = trainM_shock_fold[idxx]
    trainM_nonshock = trainM_nonshock_fold[idxx]
    testM_shock = testM_shock_fold[idxx]
    testM_nonshock = testM_nonshock_fold[idxx]
    
    trainLabels = list(np.ones(len(trainM_shock)))
    trainLabels.extend(np.zeros(len(trainM_nonshock)))
    testLabels = list(np.ones(len(testM_shock)))
    testLabels.extend(np.zeros(len(testM_nonshock)))
    
    df1 = pd.DataFrame(data=binaryMatrix_train_all[idxx])
    df1.insert(loc=0, column='LABEL', value=trainLabels)
    
    df2 = pd.DataFrame(data=binaryMatrix_test_all[idxx])
    df2.insert(loc=0, column='LABEL', value=testLabels)
    
    shock_pattern_list = shock_list_fold[idxx]
    nonshock_pattern_list = nonshock_list_fold[idxx]
    
    
    train_csv_name = 'extraction_train_fold'+str(idxx)+'.csv'
    test_csv_name = 'extraction_test_fold'+str(idxx)+'.csv'
    df1.to_csv(train_csv_name, index=False)
    df2.to_csv(test_csv_name, index=False)
    
    shock_txt_name = 'shock_pattern_list_fold'+str(idxx)+'.txt'
    nonshock_txt_name = 'nonshock_pattern_list_fold'+str(idxx)+'.txt'
    with open(shock_txt_name, "wb") as fp:   #Pickling
        pickle.dump(shock_pattern_list, fp) 
       
    with open(nonshock_txt_name, "wb") as fp:   #Pickling
        pickle.dump(nonshock_pattern_list, fp)    

# ###############################################################################
# # PartE. Preprocessing Before Extracting MPTP and maixmal RTPs (All Data) #####
# ###############################################################################
Labels = list(np.ones(len(trainBinaryMatrixSet_shock)))
Labels.extend(np.zeros(len(trainBinaryMatrixSet_nonshock))) 
    
df_ALL = pd.DataFrame(data=binaryMatrix_ALL)
df_ALL.insert(loc=0, column='LABEL', value=Labels)
    
shock_pattern_list_ALL = top_pattern_shock_ALL
nonshock_pattern_list_ALL = top_pattern_nonshock_ALL
    
csv_name = 'extraction_ALL.csv'
df_ALL.to_csv(csv_name, index=False)
    
shock_txt_name = 'shock_pattern_list_ALL.txt'
nonshock_txt_name = 'nonshock_pattern_list_ALL.txt'
with open(shock_txt_name, "wb") as fp:   #Pickling
    pickle.dump(shock_pattern_list_ALL, fp) 
       
with open(nonshock_txt_name, "wb") as fp:   #Pickling
    pickle.dump(nonshock_pattern_list_ALL, fp)        
    
    
################################################################################################################################
# PartF. Save shock probability and nonshock probability for ALL data mining; used as display in paper section (not necessary) #
################################################################################################################################
ALL_shock_txt_name = 'shock_pattern_list_ALL.txt'
ALL_nonshock_txt_name = 'nonshock_pattern_list_ALL.txt'
with open(ALL_shock_txt_name, "rb") as fp:   # Unpickling
    ALL_shock_pattern_list = pickle.load(fp)  
with open(ALL_nonshock_txt_name, "rb") as fp:   # Unpickling
    ALL_nonshock_pattern_list = pickle.load(fp)  

df_binary_ALL = pd.read_csv('extraction_ALL.csv')
array_binary_ALL = df_binary_ALL.iloc[:,1:].to_numpy()

shock_support_list_ALL = []
for idx in range(array_binary_ALL.shape[1]):
    shock_support_list_ALL.append(array_binary_ALL[0:1819,idx].sum()/1819)
nonshock_support_list_ALL = []
for idx in range(array_binary_ALL.shape[1]):
    nonshock_support_list_ALL.append(array_binary_ALL[1819:,idx].sum()/2585) 
   
    
excel_file_name = 'NonContrasted_0.3_0.3_ALL_Patterns_Info.xlsx'
workbook = xlsxwriter.Workbook(os.getcwd()+'/'+excel_file_name)
all_pattern_list_ALL = []
for item in ALL_shock_pattern_list:
    all_pattern_list_ALL.append(item.describe())
for item in ALL_nonshock_pattern_list:
    all_pattern_list_ALL.append(item.describe())
        
    
save_data = list(zip(all_pattern_list_ALL,shock_support_list_ALL,nonshock_support_list_ALL))
worksheet = workbook.add_worksheet('ALL')
for row, line in enumerate(save_data):
    for col, cell in enumerate(line):
        worksheet.write(row, col, cell)
workbook.close()        


for num in range(5):
    '''
    M = binaryMatrix_train_all[num] 
    shock_support_list = []
    for idx in range(M.shape[1]):
        shock_support_list.append(M[0:,idx].sum()/1638)
    nonshock_support_list = []
    for idx in range(M.shape[1]):
        nonshock_support_list.append(M[1638:,idx].sum()/2327)   
    '''
    trainM_shock = trainM_shock_fold[idxx]
    trainM_nonshock = trainM_nonshock_fold[idxx]
    testM_shock = testM_shock_fold[idxx]
    testM_nonshock = testM_nonshock_fold[idxx]
    
    M = binaryMatrix_train_all[num] 
    shock_support_list = []
    for idx in range(M.shape[1]):
        shock_support_list.append(M[0:len(trainM_shock),idx].sum()/len(trainM_shock))
    nonshock_support_list = []
    for idx in range(M.shape[1]):
        nonshock_support_list.append(M[len(trainM_shock):(len(trainM_shock)+len(trainM_nonshock)),idx].sum()/num_control) 

        
    
    excel_file_name = 'NonContrasted_0.3_0.3_Fold'+str(num)+'_Patterns_Info.xlsx'
    workbook = xlsxwriter.Workbook(os.getcwd()+'/'+excel_file_name)
    all_pattern_list_fold = []
    for item in shock_list_fold[num]:
        all_pattern_list_fold.append(item.describe())
    for item in nonshock_list_fold[num]:
        all_pattern_list_fold.append(item.describe())
        
    
    save_data = list(zip(all_pattern_list_fold,shock_support_list,nonshock_support_list))
    worksheet = workbook.add_worksheet('Fold'+str(num))
    for row, line in enumerate(save_data):
        for col, cell in enumerate(line):
            worksheet.write(row, col, cell)
            
    save_data = list(zip(all_pattern_list_fold,shock_support_list,nonshock_support_list))
    test = pd.DataFrame(0, index=range(len(save_data)), columns=range(3))
    for row, line in enumerate(save_data):
        for col, cell in enumerate(line):
            test.iloc[row,col] = cell
            #print(col)
            #print(cell)
            #worksheet.write(row, col, cell)
    print(test)
    workbook.close() 








