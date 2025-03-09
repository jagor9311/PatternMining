import pandas as pd
import numpy as np
import sys
from enum import Enum
import math



Features = ['condition_concept_name','drug_concept_name','device_concept_name']



class Lab_Values(Enum):
    VL = 1
    L = 2
    N = 3
    H = 4
    VH = 5
class CCHS_Location_Values(Enum):
    ED = 1
    NURSE = 2
    ICU =3
    STEPDN = 4
class Lab_Features(Enum):
    condition_concept_name = 1
    drug_concept_name = 2
    device_concept_name = 3
    '''
    RespiratoryRate = 4
    Temperature = 5
    PulseOx = 6
    BUN = 7
    Procalcitonin = 8
    WBC = 9
    Bands = 10
    Lactate = 11
    Platelet = 12
    Creatinine = 13
    MAP = 14
    BiliRubin = 15
    CReactiveProtein =16
    SedRate = 17
    OxygenFlow = 19
    '''
class Binary_Features(Enum):
    InfectionFlag=1
    InflammationFlag=2
    OrganFailure=3


class State:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
    def describe(self):
        return "(" + self.feature + "," + str(self.value) + ")"
    def __eq__(self, other):
        if self.feature == other.feature and self.value == other.value:
            return True
        return False
    def __hash__(self):
        return hash((self.feature,self.value))

class StateInterval:
    def __init__(self, feature, value, start, end):
        self.feature = feature
        self.value = value
        self.start = start
        self.end = end
    def __gt__(self, state2):
        return self.start > state2.start
    def describe(self):
        return "(" + self.feature + "," + str(self.value) + "," + str(self.start) + "," + str(self.end) + ")"
    def find_relation(self, s2):
        if self.end < s2.start:
            return 'b'
        elif self.start <= s2.start and s2.start <= self.end:
            return 'c'
        else:
            return 'e'        


def abstraction_alphabet(f1, f0, f):                            #abstracts the values for a feature of the whole data    
    if f == "SystolicBP":
        VL_range = 60
        L_range = 90
        N_range = 140
        H_range = 180
    if f == "DiastolicBP":
        VL_range = 40
        L_range = 60
        N_range = 90
        H_range = 120
    if f == "HeartRate":
        VL_range = 50
        L_range = 70
        N_range = 100
        H_range = 120
    if f == "RespiratoryRate":
        VL_range = 8
        L_range = 12
        N_range = 20
        H_range = 24
    if f == "Temperature":
        VL_range = 35
        L_range = 36.1
        N_range = 37.2
        H_range = 39
    if f == "PulseOx":
        VL_range = 90
        L_range = 95
        N_range = 100
        H_range = 101
    if f == "BUN":
        VL_range = 4
        L_range = 7
        N_range = 20
        H_range = 30
    if f == "Procalcitonin":
        VL_range = 0.07
        L_range = 0.1
        N_range = 2.0
        H_range = 10
    if f == "WBC":
        VL_range = 2.5
        L_range = 3.4
        N_range = 9.6
        H_range = 16.0
    if f == "Bands":
        VL_range = 1.0
        L_range = 2.0
        N_range = 8.0
        H_range = 14.0
    if f == "Lactate":
        VL_range = 0.3
        L_range = 0.5
        N_range = 2.2
        H_range = 3.5
    if f == "Platelet":
        VL_range = 100
        L_range = 150
        N_range = 400
        H_range = 450
    if f == "Creatinine":
        VL_range = 0.3
        L_range = 0.5
        N_range = 1.2
        H_range = 2.0
    if f == "MAP":
        VL_range = 60
        L_range = 70
        N_range = 100
        H_range = 110
    if f == "BiliRubin":
        VL_range = 0.1
        L_range = 0.2
        N_range = 1.2
        H_range = 1.8
    if f == "CReactiveProtein":
        VL_range = 0
        L_range = 0
        N_range = 10
        H_range = 50
    if f == "SedRate":
        VL_range = 0
        L_range = 0
        N_range = 30
        H_range = 50
    if f == "OxygenFlow":
        VL_range = 1.0
        L_range = 2.0
        N_range = 4.0
        H_range = 6.0 
    
    lab_values = pd.concat([f1, f0])
    VH_range = np.percentile(lab_values[np.isfinite(lab_values)],100)
    
    shock = pd.DataFrame(f1)
    shock.loc[f1<VL_range,shock.columns.values[0]] = "VL"
    shock.loc[(f1>=VL_range) & (f1<L_range),shock.columns.values[0]] = "L"
    shock.loc[(f1>=L_range) & (f1<N_range),shock.columns.values[0]] = "N"
    shock.loc[(f1>=N_range) & (f1<H_range),shock.columns.values[0]] = "H"
    shock.loc[(f1>=H_range) & (f1<=VH_range),shock.columns.values[0]] = "VH"
    
    nonshock = pd.DataFrame(f0)
    nonshock.loc[f0<VL_range,nonshock.columns.values[0]] = "VL"
    nonshock.loc[(f0>=VL_range) & (f0<L_range),nonshock.columns.values[0]] = "L"
    nonshock.loc[(f0>=L_range) & (f0<N_range),nonshock.columns.values[0]] = "N"
    nonshock.loc[(f0>=N_range) & (f0<H_range),nonshock.columns.values[0]] = "H"
    nonshock.loc[(f0>=H_range) & (f0<=VH_range),nonshock.columns.values[0]] = "VH"
    return shock, nonshock


def state_generation(abstracted_lab_values, feature):            #this function gives abstracted values for a feature for a patient and returns the state intervals generated
    state_intervals = []
    previous_value = np.nan
    state_start = np.nan
    state_end = np.nan
    for i,val in abstracted_lab_values.iterrows():
        if pd.notnull(val[feature]) and pd.isnull(previous_value):
            previous_value = val[feature]
            state_start = val['days_from_first_event']
            state_end = val['days_from_first_event']
        elif pd.notnull(val[feature]) and (val[feature]==previous_value):
            state_end = val['days_from_first_event']
        elif pd.notnull(val[feature]) and (val[feature]!=previous_value):
            state_intervals.append(StateInterval(feature,previous_value,state_start,state_end))
            previous_value = val[feature]
            state_start = val['days_from_first_event']
            state_end = val['days_from_first_event']
    if pd.notnull(previous_value) and pd.notnull(state_end) and pd.notnull(state_start):
        state_intervals.append(StateInterval(feature,previous_value,state_start,state_end))
    return state_intervals


def MultivariateStateSequence(sequence_data):                    #this function gives a sequence of data (for one patient) and returns the MSS
    MSS = []
    for f in Features:
        MSS.extend(state_generation(sequence_data, f))
    '''
    for f in Binary_Features:
        MSS.extend(state_generation(sequence_data, f.name))
    MSS.extend(state_generation(sequence_data, 'CurrentLocationTypeCode'))
    MSS.sort(key=lambda x: x.start)
    '''
    return MSS


def state_find_matches(mss, state, fi):                            #Find the index of state intervals in an MSS with same feature and value of state
    match = []
    for i in range (fi, len(mss)):
        if state.feature == mss[i].feature and state.value == mss[i].value:
            match.append(i)
    return match


# ---- A recursive function that determines whether a sequence contains a pattern or not
def MSS_contains_Pattern(mss, p, i, fi, prev_match):            
    if i >= len(p.states):
        return True, prev_match
    same_state_index = state_find_matches(mss, p.states[i], fi)
    for fi in same_state_index:
        flag = True
        for pv in range(0,len(prev_match)):
            if prev_match[pv].find_relation(mss[fi]) != p.relation[pv][i]:
                flag = False
                break
        if flag:
            prev_match.append(mss[fi])
            contains, seq = MSS_contains_Pattern(mss, p, i+1, 0, prev_match)
            if contains:
                return True, seq
            else:
                del prev_match[-1]
    return False, np.nan


def recent_state_interval(max_val, mss, j, g):            #Determines whether a state interval is recent or not
    if max_val - mss[j].end < g:
        return True
    return False


def get_index_in_sequence(mss, e):                #Finds the index of a state interval in a MSS
    for i in range(0,len(mss)):
        if mss[i] == e:
            return i
    return -1


def sequences_containing_state(RTPlist, new_s):
    p_RTPlist = []
    for z in RTPlist:
        for e in z:
            if e.feature == new_s.feature and e.value == new_s.value:
                p_RTPlist.append(z)
                break
    return p_RTPlist

#find_all_frequent_states(D, support,featureValues)
def find_all_frequent_states(D, support,featureValues):
    freq_states = []
    for i in range(len(Features)):
        f = Features[i]
        for j in range(len(featureValues[i])):
            v = featureValues[i][j]
            state = State(f,v)
            if len(sequences_containing_state(D, state)) >= support:
                freq_states.append(state)
    '''
    for f in Binary_Features:
        for v in (0,1):
            state = State(f.name,v)
            if len(sequences_containing_state(D, state)) >= support:
                freq_states.append(state)
    
    Location_Values = CCHS_Location_Values
    for v in Location_Values:
        state = State('CurrentLocationTypeCode', v.name)
        if len(sequences_containing_state(D, state)) >= support:
            freq_states.append(state)
    '''
    return freq_states 


def sequences_containing_state_NEW(RTPlist, new_s, kRTP_idx_list, MSS_idx_list):
    p_RTPlist = []
    new_kRTP_idx_list = []
    new_MSS_idx_list = []
    for idx in range(0, len(RTPlist)):
        z = RTPlist[idx]
        for e in z:
            if e.feature == new_s.feature and e.value == new_s.value:
                p_RTPlist.append(z)
                new_kRTP_idx_list.append(kRTP_idx_list[idx])
                new_MSS_idx_list.append(MSS_idx_list[idx])
                break
    return p_RTPlist, new_kRTP_idx_list, new_MSS_idx_list


