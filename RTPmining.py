import pandas as pd
import numpy as np
import math
import itertools

import TemporalAbstraction


class TemporalPattern:
    def __init__(self, states, relation, RTPlist, p_RTPlist, end):
        self.states = states
        self.relation = relation
        self.RTPlist = RTPlist
        self.p_RTPlist = p_RTPlist
        self.end = end
    def describe(self):
        string = ""
        for s in self.states:
            string = string + s.describe()          
        array_str = ""
        for row in self.relation:
            for e in row:
                array_str = array_str + e +" "
        string = string + array_str
        return string
    def __eq__(self, other):
        if len(self.states) != len(other.states):
            return False
        if set(self.states) != set(other.states):
            return False
        k = len(self.states)
        mapping = {}
        for i in range(k):
            indices = [j for j in range(k) if self.states[i]==other.states[j]]
            for idx in indices:
                if idx not in mapping.values():
                    mapping[i] = idx
                    break
            if i not in mapping:
                return False
        for i in range(k):
            for j in range(i+1,k):
                if self.relation[i][j] != other.relation[min(mapping[i],mapping[j])][max(mapping[i],mapping[j])]:
                    return False
        return True
    def __ne__(self, other):
        return (not self.__eq__(other))    
    def __hash__(self):
        return hash((self.states,self.relation))


def extend_backward(RTP,newState):
    states = RTP.states
    relations = RTP.relation
    ends = RTP.end
    k = len(states)
    sPrime = []
    sPrime.append(newState)
    sPrime[1:] = states
    rPrime = []
    row1 = []
    row1.append("o")
    for i in range(1,(k+1)):
        row1.append("b")
    rPrime.append(row1)
    for i in range(0,k):
        row = ["o"]
        row[1:] = relations[i]
        rPrime.append(row)
    pPrime = TemporalPattern(sPrime,rPrime,[],[],[ends])
    newRelation = []
    for i in range(0,len(pPrime.states)):
        newRelation.append(list(pPrime.relation[i]))
    C = [TemporalPattern(pPrime.states[:],newRelation,[],[],[ends])]
    for i in range(1,k+1):
        if sPrime[0].feature == sPrime[i].feature:
            break
        else:
            rPrime[0][i] = "c"
            pPrime = TemporalPattern(sPrime,rPrime,[],[],[ends])
            #for i in range(0,len(pPrime.states)):
                #print pPrime.states[i].describe()
            if not any((x == pPrime) for x in C):
                newRelation = []
                for i in range(0,k+1):
                    newRelation.append(list(pPrime.relation[i]))
                C.append(TemporalPattern(pPrime.states[:],newRelation,[],[],[ends]))
    return C


def recent_temporal_pattern_NEW(mss, p, g, g_int, idx_list):            #Determines whether a pattern is RTP or not based on interval
    same_state_index = TemporalAbstraction.state_find_matches(mss, p.states[0], 0) 
    all_mapping = []
    all_idx_mapping = []
    for add_idx in same_state_index:
        for sure_idx in idx_list:
            mapping = []
            idx_mapping = []
            for compare_num in range(1,len(p.states)):
                select_idx = sure_idx[compare_num-1]
                if mss[add_idx].find_relation(mss[select_idx]) == p.relation[0][compare_num]:
                    flag = True
                else:
                    flag = False
                    break
            if flag:
                mapping.append(mss[add_idx])
                idx_mapping.append(add_idx)
                for item in sure_idx:
                    mapping.append(mss[item]) #[first,...,last]
                    idx_mapping.append(item) #[first,...,last]
                all_mapping.append(mapping)
                all_idx_mapping.append(idx_mapping)
    #print(len(all_mapping)) #check
    #print(len(all_idx_mapping)) #check
    final_idx_mapping_list = []
    for num in range(0, len(all_mapping)):
        check_bool = True
        current_mapping = all_mapping[num]
        current_mapping_idx = all_idx_mapping[num]
        #print(TemporalAbstraction.get_index_in_sequence(mss, current_mapping[len(p.states)-1]))
        #print(current_mapping_idx[0])
        # The first check must be true
        max_Z_val = max([item.end for item in mss])
        if not TemporalAbstraction.recent_state_interval(max_Z_val, mss, TemporalAbstraction.get_index_in_sequence(mss, current_mapping[len(p.states)-1]), g):
            check_bool = False
        for i in range(0,len(current_mapping)-1):
            if current_mapping[i+1].start - current_mapping[i].end > g_int: 
                check_bool = False
        if check_bool:
            final_idx_mapping_list.append(current_mapping_idx)
    if len(final_idx_mapping_list) != 0:
        return True, final_idx_mapping_list, np.nan
    
    return False, np.nan, np.nan

    
def RTP_support(P, g, g_int, passing_list, MSS_idx_passing_list):                        #calculating support of a recent temporal pattern using DEFINITION 6
    RTPlist = []
    new_idx_list_for_the_RTP = []
    new_MSS_idx_list_for_the_RTP = []
    endlist = []
    for idx in range(0, len(P.p_RTPlist)):
        Z = P.p_RTPlist[idx]
        passing_idx_list = passing_list[idx]
        #print(len(Z))
        #print(len(passing_idx_list))
        bool_res, new_idx_list_for_Z, endtimeval = recent_temporal_pattern_NEW(Z, P, g, g_int, passing_idx_list)
        if bool_res:
            RTPlist.append(Z)
            new_idx_list_for_the_RTP.append(new_idx_list_for_Z)
            new_MSS_idx_list_for_the_RTP.append(MSS_idx_passing_list[idx])
            endlist.append(endtimeval)
    return RTPlist, new_idx_list_for_the_RTP, new_MSS_idx_list_for_the_RTP, endlist

def counting_phase(candidates, g, support, g_int, kRTP, candidates_kRTP_idx_list, candidates_MSS_idx_list):
    kRTP_new = []
    kRTP_idx_list = []
    MSS_idx_list = []
    for num in range(0,len(candidates)):
        C = candidates[num]
        passing_list = candidates_kRTP_idx_list[num]
        MSS_idx_passing_list = candidates_MSS_idx_list[num]
        C.RTPlist,idx_list_for_C, MSS_idx_list_for_C, C.end = RTP_support(C, g, g_int, passing_list, MSS_idx_passing_list)
        #print(len(C.RTPlist)) #check
        #print(len(idx_list_for_C)) #check
        if len(C.RTPlist) >= support:
            kRTP_new.append(C)
            kRTP_idx_list.append(idx_list_for_C)
            MSS_idx_list.append(MSS_idx_list_for_C)
            ### append the create large 3D list
    return kRTP_new, kRTP_idx_list, MSS_idx_list

#candidate_generation(D, kRTP, freq_states, g, support, kRTP_idx_list, MSS_idx_list)
#p_states = freq_states
def candidate_generation(D, kRTP, p_states, g, support, kRTP_idx_list, MSS_idx_list):
    candidates = []
    candidates_kRTP_idx_list = []
    candidates_MSS_idx_list = []
    for idx in range(len(kRTP)):
        print(idx)
        p = kRTP[idx]
        for s in p_states:
            C = extend_backward(p, s)
            for q in range(0, len(C)):
                change_list = kRTP_idx_list[idx]
                MSS_change_list = MSS_idx_list[idx]
                # for idx in range(0, len(kRTP)):
                #     re = C[q].relation
                #     check_re = [item[1:] for item in re[1:]]
                #     if kRTP[idx].states == C[q].states[1:] and kRTP[idx].relation == check_re:
                #         change_list = kRTP_idx_list[idx]
                #         MSS_change_list = MSS_idx_list[idx]
                #         break
                C[q].p_RTPlist, new_kRTP_idx_list, new_MSS_idx_list= TemporalAbstraction.sequences_containing_state_NEW(p.RTPlist, s, change_list, MSS_change_list)
                if len(C[q].p_RTPlist) >= support:
                    if C[q] not in candidates:
                    #if not any((x == C[q]) for x in candidates):
                        candidates.append(C[q])
                        candidates_kRTP_idx_list.append(new_kRTP_idx_list)
                        candidates_MSS_idx_list.append(new_MSS_idx_list)
    return candidates, candidates_kRTP_idx_list, candidates_MSS_idx_list

#RTPmining.pattern_mining(allShockMSS, g, g_int, sup_developed*len(allShockMSS))
#D = allShockMSS
#support = sup_developed*len(allShockMSS)
def pattern_mining(D, g, g_int, support,featureValues):
    one_RTP = []
    freq_states = TemporalAbstraction.find_all_frequent_states(D,support,featureValues)
    print("number of frequent states are:", len(freq_states))
    end_idx_list_AllOneStates = []
    MSS_idx_list_AllOneStates = []
    
    max_end_time_list = []
    for Z in D:
        max_end_time_list.append(max([item.end for item in Z]))
    
    # Find numbers of recent(RTP2) and frequent(RTP4) temporal patterns among frequent patterns (29/45)
    for s in freq_states:
        new_pattern = TemporalPattern([s],[['o']],[],[],[])
        RTPlist = []
        endlist = []
        end_idx_list_OneStates= []
        MSS_idx_list_OneStates= []
        for new_idx in range(len(D)):
            Z = D[new_idx]
            Z_endTime = max_end_time_list[new_idx]
            interval_matches = TemporalAbstraction.state_find_matches(Z, s, 0)
            if len(interval_matches) != 0:
                et_array = []
                if TemporalAbstraction.recent_state_interval(Z_endTime, Z, max(interval_matches), g):
                    end_idx = []
                    RTPlist.append(Z)
                    for item in interval_matches:
                        if TemporalAbstraction.recent_state_interval(Z_endTime, Z, item, g):
                            et_array.append(Z[item].end)
                            end_idx.append([item]) #End idx which is recent list per MSS per states
                    end_idx_list_OneStates.append(end_idx)
                    MSS_idx_list_OneStates.append(new_idx)
                endlist.append(et_array)
            else:
                endlist.append(None)
        if len(RTPlist) >= support:
            new_pattern.RTPlist = RTPlist
            new_pattern.p_RTPlist = RTPlist
            new_pattern.end = endlist
            one_RTP.append(new_pattern)
            end_idx_list_AllOneStates.append(end_idx_list_OneStates)
            MSS_idx_list_AllOneStates.append(MSS_idx_list_OneStates)

    print("the number of one-RTPs:", len(one_RTP))
    K = max(len(z) for z in D) # maximum numbers of state interval in a MSS (assume all form a pattern)
    kRTP = one_RTP
    kRTP_idx_list = end_idx_list_AllOneStates
    MSS_idx_list = MSS_idx_list_AllOneStates
    Omega = []
    Omega.extend(one_RTP)
    kRTP_idx_list_final_4D = []
    kRTP_idx_list_final_4D.append(kRTP_idx_list)
    MSS_idx_list_final_4D = []
    MSS_idx_list_final_4D.append(MSS_idx_list)
    count = 0
    for k in range(1, K+1):
        if k>= count*100: 
            print(k)
            count+=1
        candidates, candidates_kRTP_idx_list, candidates_MSS_idx_list= candidate_generation(D, kRTP, freq_states, g, support, kRTP_idx_list, MSS_idx_list)
        print ("------------------------length of the candidates for", k+1, "pattern is:", len(candidates))
        kRTP, kRTP_idx_list, MSS_idx_list = counting_phase(candidates, g, support, g_int, kRTP, candidates_kRTP_idx_list, candidates_MSS_idx_list)
        print ("------------------------length of the kRTP for", k+1, "pattern is:", len(kRTP))
        print(len(kRTP_idx_list))
        if len(kRTP) == 0:
            break
        Omega.extend(kRTP)
        kRTP_idx_list_final_4D.append(kRTP_idx_list)
        MSS_idx_list_final_4D.append(MSS_idx_list)
    
    print ("number of all patterns found:", len(Omega))
    return Omega, kRTP_idx_list_final_4D, MSS_idx_list_final_4D


def pattern_mss_opposite_match(oppositeD, passing_patterns, g, g_int):
    final_end_idx_return = [None] * len(passing_patterns)
    final_MSS_idx_return = [None] * len(passing_patterns)
    
    max_end_time_list = []
    for Z in oppositeD:
        max_end_time_list.append(max([item.end for item in Z]))
        
    for idx in range(0, len(passing_patterns)):
        if len(passing_patterns[idx].states) == 1:
            end_idx_list_OneStates= []
            MSS_idx_list_OneStates= []
            for new_idx in range(len(oppositeD)):
                Z = oppositeD[new_idx]
                states_seq = passing_patterns[idx].states
                interval_matches = TemporalAbstraction.state_find_matches(Z, states_seq[0], 0)
                if len(interval_matches) != 0:
                    if TemporalAbstraction.recent_state_interval(max_end_time_list[new_idx], Z, max(interval_matches), g):
                        end_idx = []
                        for item in interval_matches:
                            if TemporalAbstraction.recent_state_interval(max_end_time_list[new_idx], Z, item, g):
                                end_idx.append([item])
                        end_idx_list_OneStates.append(end_idx)
                        MSS_idx = oppositeD.index(Z)
                        MSS_idx_list_OneStates.append(MSS_idx)
            final_end_idx_return[idx] = end_idx_list_OneStates
            final_MSS_idx_return[idx] = MSS_idx_list_OneStates
        else:
            p = passing_patterns[idx]
            end_idx_list_OneStates= []
            MSS_idx_list_OneStates= []
            for new_idx in range(len(oppositeD)):
                Z = oppositeD[new_idx]
                big_match_list = []
                for num in range(0, len(p.states)):
                    interval_matches = TemporalAbstraction.state_find_matches(Z, p.states[num], 0)
                    big_match_list.append(interval_matches)
                Flag = True
                for item in big_match_list:
                    if len(item) == 0:
                        Flag = False
                        break
                if Flag:
                    all_possible_patterns = list(itertools.product(*big_match_list))
                    end_idx = []
                    for item in all_possible_patterns:
                        new_flag = True
                        for idx1 in range(0, len(item)-1):
                            for idx2 in range(idx1+1, len(item)):
                                select_idx1 = item[idx1]
                                select_idx2 = item[idx2]
                                if Z[select_idx1].find_relation(Z[select_idx2]) != p.relation[idx1][idx2]:
                                    new_flag = False
                        if new_flag:
                            check_bool = True
                            if not TemporalAbstraction.recent_state_interval(max_end_time_list[new_idx], Z, item[-1], g):
                                check_bool = False
                            for i in range(0,len(item)-1):
                                mss_idx1 = item[i+1]
                                mss_idx2 = item[i]
                                if Z[mss_idx1].start - Z[mss_idx2].end > g_int: 
                                    check_bool = False
                            if check_bool:
                                end_idx.append(list(item))
                    if len(end_idx) != 0:
                        end_idx_list_OneStates.append(end_idx)
                        MSS_idx = oppositeD.index(Z)
                        MSS_idx_list_OneStates.append(MSS_idx)
            
            final_end_idx_return[idx] = end_idx_list_OneStates
            final_MSS_idx_return[idx] = MSS_idx_list_OneStates
    return final_end_idx_return, final_MSS_idx_return
            
                
                                
                            
                            
                    
                        
                    
                    
                    
                    
                        
                        
                    
                
            
                        
                
