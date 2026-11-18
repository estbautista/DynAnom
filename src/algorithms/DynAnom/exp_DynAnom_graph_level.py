# Adapted from DynAnom by Guo, Zhou, and Skiena (KDD 2022).
# Modified for window aggregation, experiment configuration, and score export.
# License: GPL-3.0. See LICENSE and MODIFICATIONS.md.

# path
import sys
import csv
import math
import os
import functools
from os.path import join as os_join 
from os.path import dirname as up_dir
# append relative path
# current:  PATH-TO/kdd22-anomaly/src/algorithms/DynAnom
project_root = functools.reduce(lambda x,y: up_dir(x), range(4) ,os.path.abspath(__file__) )
sys.path.append(os_join(project_root,'src'))
sys.path.append(os_join(project_root,'src',"algorithms", "DynAnom"))
import argparse
import pprint
import scipy.stats as stats


# home-made lib
import graph_data_utils
import DynamicPPE

# computation
import numpy as np
import networkx as nx
from time import process_time as ptime
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm as tqdm
from networkx.algorithms.components import number_connected_components as num_cc
from networkx.algorithms.components import is_weakly_connected as is_wk_cc
import pickle
import numba
from scipy.sparse.linalg import norm as sp_norm

import lsmanip
import lsmanip.io as lio
import tempfile
from pathlib import Path


## plot related
import matplotlib
matplotlib.rcParams['font.family'] = "serif"
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize=2)
DISPLAY_SNAPSHOT_INVERVAL = 2


from config import get_config, set_dir


def l_p_err_nnz_row(sp_prev_appr,  sp_curt_appr, shared_nodes, l=2):
    # get l1-error of shared_nodes rows of sparse matrices. 
    if len(shared_nodes) == 0: 
        return np.array(0.0)

    a = sp_prev_appr[shared_nodes]
    b = sp_curt_appr[shared_nodes]
    if isinstance(a, csr_matrix):
        _dis = sp_norm(a-b,  axis=1, ord =l)
    else:
        _dis = np.linalg.norm(a-b,  axis=1, ord =l)
    return _dis

def _kendalltau(a,b):# approximation
    return stats.kendalltau(a,b,method ='asymptotic')

kendalltau_func = np.vectorize(_kendalltau, signature='(n),(n)->(),()')
def kendalltau_arr_nnz_row(sp_prev_appr,  sp_curt_appr, shared_nodes, total_nodes):
    prev_ppv_dense_arr =  sp_prev_appr[shared_nodes].toarray()[:,:total_nodes]
    curt_ppv_dense_arr =  sp_curt_appr[shared_nodes].toarray()[:,:total_nodes]
    tau_arr, p_arr = kendalltau_func(prev_ppv_dense_arr , curt_ppv_dense_arr)
    return (1-tau_arr)*0.5
    
spearmanr_func = np.vectorize(stats.spearmanr, signature='(n),(n)->(),()')
def spearmanr_arr_nnz_row(sp_prev_appr,  sp_curt_appr, shared_nodes, total_nodes):
    prev_ppv_dense_arr =  sp_prev_appr[shared_nodes].toarray()[:,:total_nodes]
    curt_ppv_dense_arr =  sp_curt_appr[shared_nodes].toarray()[:,:total_nodes]
    spearman_arr, p_arr = spearmanr_func(prev_ppv_dense_arr , curt_ppv_dense_arr)
    return (1-spearman_arr)*0.5 


def accuracy_score(y_pred_set, y_true_set):
    fp = 0
    tp = 0
    
    for y_pred in y_pred_set:
        if y_pred in y_true_set:
            tp += 1
        else:
            fp += 1
    precision = tp / (len(y_pred_set))
    recall = tp / (len(y_true_set))
    if precision + recall == 0:
        f1 = 0
    else:
        f1= 2*(precision*recall)/ (precision+recall)
    return precision, recall, f1

def test_graph_stream(  path, dataset_name, timeStep, initSS , \
                        make_directed_graph, make_multi_graph, 
                        alpha, beta, epsilon, track_mode, top_index, dim, \
                        validate_appr,selective_push_threshold,  detect_anomaly, write_snapshot_ppvs):
    # for output file format
    data_config_str = f'dataset_{dataset_name}-initSS_{initSS}-timeStep_{timeStep}'
    algo_config_str = f'DynAnomPy-alpha_{alpha:.2e}-epsilon_{epsilon:.2e}-track_mode_{track_mode}-dim-{dim}-top_index_{top_index}-push_threshold-{selective_push_threshold:.2e}-make_directed_graph_{make_directed_graph}-make_multi_graph_{make_multi_graph}'
    print (f'data: {data_config_str}')
    print (f'algo: {algo_config_str}')

    ################################################################
    ####    Graph snapshots construcution
    ################################################################
    # Some edge stream may only have 10 edges,  but results in 100 snapshots.
    # In this case, a great amount of snapshots are equal since no edge event happens between many of them.
    
    #timeStep = 60 #time interval of one snapshot, e.g.: 60 mins; or 60 sec
    n, m, timeNum, nodes, edges, snapshots = graph_data_utils.read_data(path, timeStep)


    attackLimit = 50 # threshold for attack snapshot anomaly
    Inject_param = False 
    injectNum = 50
    injectSize = 70
    numSS = (timeNum-1)//timeStep
    testNum = numSS - initSS

    attack = np.zeros(testNum+1) # the groud-truth of each snapshot

    print (f'#nodes:{n} , #edges:{m}, timeNum:{timeNum}, total snapshots: {numSS}, init snapshots: {initSS}, effective snapshots :{len(snapshots)}')

    # choose which snapshots should be injected with random edges as anomaly 
    # it will add new effective snapshots (step).
    if Inject_param is True:
        snapshots, injectSS = graph_data_utils.inject_snapshot(injectNum, initSS, testNum, snapshots)
        print (f'#injectSS:{len(injectSS)}, #snapshots: {len(snapshots)}')

    # different data structure of graph 
    eg = 0
    injected = 0
    current_m = 0

    ################################################################
    ####    Algorithm-specific parameters 
    ################################################################
    
    alpha = alpha #0.15
    beta = beta #0.0
    epsilon = epsilon# 1e-6
    validate_appr = validate_appr #False
    detect_anomaly = detect_anomaly# False
    write_snapshot_ppvs = write_snapshot_ppvs# False
    #make_directed_graph, make_multi_graph
    estimator = DynamicPPE.GraphPPREstimator(n=n, alpha= alpha, epsilon = epsilon, beta = beta, 
                                            selective_push_threshold = selective_push_threshold,
                                            is_directed_graph = make_directed_graph,
                                            is_multi_graph = make_multi_graph, 
                                            update_networkx = validate_appr)
    
    # query_node_desc_set -> the tracking nodes
    query_node_desc_set = set([])
    #query_node_desc_set = [48, 13] # for testing
    
    # For validate_appr 
    snapshot_ppv_l1_err = []
    prev_appr = None
    
    dict_anomaly_scores = {}
    def __assign_anomaly_score(key, crt_snapshot, anom_score):
        if key not in  dict_anomaly_scores.keys():
            dict_anomaly_scores[key] = np.zeros(testNum+1)
        dict_anomaly_scores[key][crt_snapshot - initSS] = anom_score

    # Anomaly score calculation
    def __anom_v1(prev_appr, sp_appr, crt_snapshot, shared_nodes, union_nodes, curt_total_nodes):
        # detect graph-level anomaly 
        # use the first order ppv changes of nodes as anomaly scores.

        l2_sum_arr = l_p_err_nnz_row(prev_appr['sp_ppv'],  sp_appr['sp_ppv'], np.array(list(shared_nodes)), l = 2)
        """
        l1_sum_arr = l_p_err_nnz_row(prev_appr['sp_ppv'],  sp_appr['sp_ppv'], np.array(list(shared_nodes)), l = 1)
        l2_sum_arr = l_p_err_nnz_row(prev_appr['sp_ppv'],  sp_appr['sp_ppv'], np.array(list(shared_nodes)), l = 2)
        
        l1_sum_dynppe_arr = l_p_err_nnz_row(prev_appr['dynppe_emb'],  sp_appr['dynppe_emb'], np.array(list(shared_nodes)), l = 1)
        l2_sum_dynppe_arr = l_p_err_nnz_row(prev_appr['dynppe_emb'],  sp_appr['dynppe_emb'], np.array(list(shared_nodes)), l = 2)
        """

        # use different anomaly detection functions here.
        if crt_snapshot >= initSS: # for those testing snapshot
            attack[crt_snapshot - initSS] = (attackNum >= attackLimit) # ground truth
            __assign_anomaly_score('DynAnomScore(l2-sum)', crt_snapshot, l2_sum_arr.sum() )
            """
            __assign_anomaly_score('DynAnomScore(l1-sum)', crt_snapshot, l1_sum_arr.sum() )
            __assign_anomaly_score('DynAnomScore(l1-mean)', crt_snapshot, l1_sum_arr.mean() )
            __assign_anomaly_score('DynAnomScore(l1-median)', crt_snapshot, np.median(l1_sum_arr) )
            __assign_anomaly_score('DynAnomScore(l1-max)', crt_snapshot, l1_sum_arr.max() )
            __assign_anomaly_score('DynAnomScore(l1-min)', crt_snapshot, l1_sum_arr.min() )
            top10_mean_idx = np.argpartition(l1_sum_arr, -10)[-10:]
            top10_mean = l1_sum_arr[top10_mean_idx].mean()
            __assign_anomaly_score('DynAnomScore(l1-max-top10)', crt_snapshot, top10_mean )

            __assign_anomaly_score('DynAnomScore(l2-sum)', crt_snapshot, l2_sum_arr.sum() )
            __assign_anomaly_score('DynAnomScore(l2-mean)', crt_snapshot, l2_sum_arr.mean() )
            __assign_anomaly_score('DynAnomScore(l2-median)', crt_snapshot, np.median(l2_sum_arr) )
            __assign_anomaly_score('DynAnomScore(l2-max)', crt_snapshot, l2_sum_arr.max() )
            __assign_anomaly_score('DynAnomScore(l2-min)', crt_snapshot, l2_sum_arr.min() )
            top10_mean_idx = np.argpartition(l2_sum_arr, -10)[-10:]
            top10_mean = l2_sum_arr[top10_mean_idx].mean()
            __assign_anomaly_score('DynAnomScore(l2-max-top10)', crt_snapshot, top10_mean )
            
            __assign_anomaly_score('DynAnomScore(l1-sum-ppe)', crt_snapshot, l1_sum_dynppe_arr.sum() )
            __assign_anomaly_score('DynAnomScore(l1-mean-ppe)', crt_snapshot, l1_sum_dynppe_arr.mean() )
            __assign_anomaly_score('DynAnomScore(l1-median-ppe)', crt_snapshot, np.median(l1_sum_dynppe_arr) )
            __assign_anomaly_score('DynAnomScore(l1-max-ppe)', crt_snapshot, l1_sum_dynppe_arr.max() )
            __assign_anomaly_score('DynAnomScore(l1-min-ppe)', crt_snapshot, l1_sum_dynppe_arr.min() )
            top10_mean_idx = np.argpartition(l1_sum_dynppe_arr, -10)[-10:]
            top10_mean = l1_sum_dynppe_arr[top10_mean_idx].mean()
            __assign_anomaly_score('DynAnomScore(l1-max-top10-ppe)', crt_snapshot, top10_mean )
            
            __assign_anomaly_score('DynAnomScore(l2-sum-ppe)', crt_snapshot, l2_sum_dynppe_arr.sum() )
            __assign_anomaly_score('DynAnomScore(l2-mean-ppe)', crt_snapshot, l2_sum_dynppe_arr.mean() )
            __assign_anomaly_score('DynAnomScore(l2-median-ppe)', crt_snapshot, np.median(l2_sum_dynppe_arr) )
            __assign_anomaly_score('DynAnomScore(l2-max-ppe)', crt_snapshot, l2_sum_dynppe_arr.max() )
            __assign_anomaly_score('DynAnomScore(l2-min-ppe)', crt_snapshot, l2_sum_dynppe_arr.min() )
            top10_mean_idx = np.argpartition(l2_sum_dynppe_arr, -10)[-10:]
            top10_mean = l2_sum_dynppe_arr[top10_mean_idx].mean()
            __assign_anomaly_score('DynAnomScore(l2-max-top10-ppe)', crt_snapshot, top10_mean )
            """


    ################################################################
    ####    Iterate snapshot and calculate anomaly scores 
    ################################################################
    with tqdm(range(len(snapshots)), miniters=100) as t:
        for loop_i, ss in enumerate(t):
            # count the total number of attack edges in this snapshot, 
            # if > attackLimit, then this snapshot is anomalous
            attackNum = 0
            
            # newly added edges
            new_edges_dict = {}
            
            # Incrementally update graph
            while eg < len(edges) and edges[eg][2] <= snapshots[ss]*timeStep:
                # If this edge happens before the time border, inject edges/update graph. 
                edge = edges[eg]
                eg += 1

                current_m += 1 # current_m records current all injected edges, including artifically augmented anomalous edges
                
                # this edge is anomalous by its label.
                if edge[3] == 1:
                    attackNum += 1 
                                                
                if edge[0] == edge[1]:
                    continue
                
                if (edge[0], edge[1]) not in  new_edges_dict.keys():
                    new_edges_dict[(edge[0], edge[1])] = [1,1, edge[3]] #time, weight, edge-label (1: attack)
                else:
                    new_edges_dict[(edge[0], edge[1])][1] += 1
                    new_edges_dict[(edge[0], edge[1])][2] += edge[3] # accumulate weights and edge-label

            # aggregate edge weights.
            new_edges = []
            for k,v in new_edges_dict.items():
                new_edges.append((k[0], k[1], v[0], v[1], v[2]))

            # record the existing nodes.
            snapshot = snapshots[ss]# snapshot is step_t, from [1, timeNum//stepSize]
            ################################################################
            ####    Algorithm-specific process for ...
            ################################################################
            # for this snapshot, add the new edges and get the APPROXIMATED ppvs of the tracked nodes.
            #return_dict = estimator.get_approx_ppv_with_new_edges_batch_multip_source(query_node_desc_set, list(new_edges))
            # _push_start = ptime()
            s_id_list, updated_s_id_list = estimator.get_approx_ppv_with_new_edges_batch_multip_source(query_node_desc_set, new_edges, bool_return_dict = False, track_mode =track_mode, top_index = top_index)
            # s_id_list contain the valid tracking nodes (those arrived before or in this snapshot)
            if loop_i%DISPLAY_SNAPSHOT_INVERVAL ==0:
                t.set_description(f'#target:{len(s_id_list)} #updtarget:{len(updated_s_id_list)} #NewEdge:{len(new_edges)} #Nodes:{len(estimator.dict_id2node.keys())} #Proc-Edge:{current_m}')


            ################################################################
            ####    Algorithm-specific process for Anomaly Detection
            ################################################################
            if detect_anomaly:
                # dictionary of sparse pp-vector
                sp_appr = {}
                new_row, new_col, new_data = DynamicPPE.get_numba_nested_dict_faster(s_id_list, estimator.dict_p_arr)
                if len(new_row) == 0:
                    _ppv_sp = csr_matrix((n, n))
                else:
                    new_row = np.hstack(new_row)
                    new_col = np.hstack(new_col)
                    new_data = np.hstack(new_data)

                    _ppv_sp = csr_matrix(
                        (new_data, (new_row, new_col)),
                        shape=(n, n)
                    )

                sp_appr['sp_ppv'] = _ppv_sp # n-by-n csr matrix, indexed by internal node id.
                sp_appr['nnz_rows'] = list(s_id_list)# list of nnz row numbers 
                sp_appr['dict_node2id'] = estimator.dict_node2id # dictionary mapping from node-desc to internal node id. 
                sp_appr['dict_id2node'] = estimator.dict_id2node # dictionary mapping from internal node id back to node-desc for interpretation. 
                sp_appr['node_degree'] = estimator.node_degree 
                sp_appr['delta_node_degree'] = estimator.delta_node_degree
                # print (f'dict load: {ptime() - _t_start}')

                """
                sp_appr['dynppe_emb'] = DynamicPPE.get_hash_embed(
                                                        estimator.hash_dim_arr, 
                                                        estimator.hash_sgn_arr, 
                                                        dim, n, _ppv_sp.indices, _ppv_sp.indptr, _ppv_sp.data)
                """
                sp_appr['anomalous_node_id'] = estimator.anomalous_node_id

                if prev_appr is not None:
                    # compare two sparse matrix, only count those nnz rows (the tracked nodes)
                    prev_nodes_ids = set(prev_appr['nnz_rows'])
                    curt_nodes_ids = set(sp_appr['nnz_rows'])
                    shared_nodes = curt_nodes_ids.intersection(prev_nodes_ids)
                    union_nodes = curt_nodes_ids.union(prev_nodes_ids)
                    # Only look for those existing nodes in both snapshot.
                    curt_total_nodes = len(sp_appr['dict_node2id'].keys())
                    ################################################################
                    ####    PPV-Based Anomaly score for each nodes
                    ################################################################
                    # node level difference for existig nodes
                    # will write result to dict_anomaly_scores
                    __anom_v1(prev_appr, sp_appr, snapshot, shared_nodes, union_nodes, curt_total_nodes) # for graph level 
                # remember to reset prev_appr
                prev_appr = sp_appr

            ################################################################
            ####    For debug and caching
            ################################################################
            if validate_appr:
                # dictionary of sparse pp-vector
                sp_appr = {}
                new_row, new_col, new_data = DynamicPPE.get_numba_nested_dict_faster(s_id_list, estimator.dict_p_arr)
                new_row =np.hstack(new_row)
                new_col =np.hstack(new_col)
                new_data =np.hstack(new_data)
                _ppv_sp = csr_matrix((new_data, (new_row, new_col)), shape=(n, n))
                sp_appr['sp_ppv'] = _ppv_sp # n-by-n csr matrix, indexed by internal node id.
                sp_appr['nnz_rows'] = s_id_list# list of nnz row numbers 

                # for this snapshot, add the new edges and get the EXACT ppvs of the tracked nodes.
                sp_eppr = {}# exact: eppr[TRACKED_NODE] =  np.array([....]) of ppv
                # get eppr one by one
                valid_query_node_desc_set = [ estimator.dict_id2node[_id] for _id in s_id_list]
                eppr_new_row, eppr_new_col, eppr_new_data = [], [], []
                for _id, query_node_desc in zip(s_id_list, valid_query_node_desc_set):
                    return_dict = estimator.get_exact_ppv_dict(query_node_desc)
                    for ind,ppr in return_dict.items():
                        eppr_new_row.append(_id)
                        eppr_new_col.append(ind)
                        eppr_new_data.append(ppr)
                _eppv_sp = csr_matrix((eppr_new_data, (eppr_new_row, eppr_new_col)), shape=(n, n))
                sp_eppr['sp_ppv'] = _eppv_sp # n-by-n csr matrix, indexed by internal node id.
                sp_eppr['nnz_rows'] = s_id_list# list of nnz row numbers 
                appr = sp_appr['sp_ppv']
                eppr = sp_eppr['sp_ppv']
                # see the l1 error between appr and eppr
                l1_err_arr = l_p_err_nnz_row(eppr, appr, s_id_list, l = 1) # scalar

                for (node_id, l1_err) in zip(s_id_list, l1_err_arr):
                    snapshot_ppv_l1_err.append(l1_err)
                    if l1_err > 0.001: # warning 
                        neighbors = estimator.g_indices[estimator.g_indptr[node_id]:estimator.g_indptr[node_id + 1]]
                        # print (f'====== #snapshot:{ss} #new_edges:{len(new_edges)} #nodes: {estimator.G.number_of_nodes()} #edges: {estimator.G.number_of_edges()} num_cc: {num_cc(estimator.G)} ======')
                        print (f'====== #snapshot:{ss} #new_edges:{len(new_edges)} #nodes: {estimator.G.number_of_nodes()} #edges: {estimator.G.number_of_edges()} is-cc: {is_wk_cc(estimator.G)} ======')
                        print ('l1-error:',l1_err)
                        print (f'node\t: {estimator.dict_id2node[node_id]}')
                        print (f'id\t: {node_id}')
                        print (f'deg.\t: {estimator.node_degree[node_id]}')
                        print (f'appr\t:', np.round(appr[node_id,:5].toarray(),3))
                        print (f'exact\t:', np.round(eppr[node_id,:5].toarray(),3))
                        print (f'neighbors: {neighbors}')
                        print ("appr is in-accurate")
            
            if write_snapshot_ppvs:
                # dictionary of sparse pp-vector
                sp_appr = {}
                new_row, new_col, new_data = DynamicPPE.get_numba_nested_dict_faster(s_id_list, estimator.dict_p_arr)
                new_row =np.hstack(new_row)
                new_col =np.hstack(new_col)
                new_data =np.hstack(new_data)
                _ppv_sp = csr_matrix((new_data, (new_row, new_col)), shape=(n, n))
                sp_appr['sp_ppv'] = _ppv_sp # n-by-n csr matrix, indexed by internal node id.
                sp_appr['nnz_rows'] = list(s_id_list)# list of nnz row numbers 
                sp_appr['dict_node2id'] = estimator.dict_node2id # dictionary mapping from node-desc to internal node id. 
                sp_appr['dict_id2node'] = estimator.dict_id2node # dictionary mapping from internal node id back to node-desc for interpretation. 
                sp_appr['node_degree'] = estimator.node_degree 
                sp_appr['delta_node_degree'] = estimator.delta_node_degree # 1*n np.arrary, record node degree change in this snapshot.
                save_path = os_join(project_root, 'tmp', data_config_str, algo_config_str)
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                if ss%100==0:
                    print (f'snapshot: {ss} -- Write ppv pickle to {save_path}')
                with open (os_join(save_path, f'{snapshot:05}.pkl'), 'wb') as f:
                    pickle.dump(sp_appr, f, protocol=pickle.HIGHEST_PROTOCOL)

    ################################################################
    ####    Evaluation: Compare the snapshots of high anomaly 
    ####                score against the ground-truth 
    ####    Metrics: 
    ####                Top-k's precision, recall, f1
    ####        
    ################################################################
    
    # write results for anomaly score of each snapshot, the format is for result ploting in visualize.ipynb
    if detect_anomaly:
        """
        save_path = os_join(project_root, 'out', data_config_str) 
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        res_file_path = os_join(save_path, f'{algo_config_str}_snapshot_anomaly_score.csv')
        
        anom_score_keys = list(dict_anomaly_scores.keys())
        """
        
        return dict_anomaly_scores['DynAnomScore(l2-sum)']
        
        """
        header_str = ','.join(anom_score_keys + ['Label']) 
        anom_val_list = [ dict_anomaly_scores[key].tolist() for key in anom_score_keys]
        anom_val_list.append(attack)
        # create result file in csv format.
        # DynAnomScore(l1-sum),DynAnomScore(l1-mean),DynAnomScore(l1-meadian),DynAnomScore(l1-max),DynAnomScore(l1-min), Label'
        # 0,0,0,0,....,1 
        # 0,0,0,0,....,1 
        # ...
        with open (res_file_path,'w') as wf: 
            print (f'{header_str}', file = wf)
            for _val_list in zip (*anom_val_list):
                # for first test snapshot
                val_str = ','.join( [ f'{_:.4f}' for _ in _val_list])
                wf.write(val_str)
                wf.write(f'\n')
        print (f'write DynAnomScores to file: {res_file_path}')
        
        # create result file in csv format.
        # topk,f1,precision,recall,#anomaly, '
        # 0,0,0,0,....,1 
        # 0,0,0,0,....,1 
        # ...
        attack_snapshot_id = set(attack.nonzero()[0].tolist()) # the attacked snapshot id.
        dict_top_score_snapshot_id = {}
        anom_score_keys = list(dict_anomaly_scores.keys())
        for anom_score_key in anom_score_keys:
            dict_top_score_snapshot_id[anom_score_key] = np.argsort(dict_anomaly_scores[anom_score_key])[::-1].tolist() # the snapshot id of highest detected anomaly score
        
        res_file_path = os_join(save_path, f'{algo_config_str}_precision_recall_score.csv')
        print (f'write DynAnomScores to file: {res_file_path}')
        
        if 'eucore' in dataset_name:
            _step = 5
        else:
            _step =50
        with open (res_file_path,'w') as wf: 
            print ( f'anomaly_score,topk,f1,precision,recall,#anomaly' , file = wf)
            #topk	f1	precision	recall	#anomaly
            for key in dict_top_score_snapshot_id.keys():
                for i in range(1, 17): # top 50, 100, ..., 500
                    topk = _step*i
                    y_true_set = set(attack_snapshot_id)
                    y_pred_set = set(dict_top_score_snapshot_id[key][:topk])
                    precision, recall, f1 = accuracy_score(y_pred_set, y_true_set)
                    print (f'{key},{topk},{f1:.4f},{precision:.4f},{recall:.4f},{len(y_true_set)}', file = wf)
        """

if __name__ == '__main__':
    
    # Load config
    config = get_config()

    # Path related
    dataset = config['paths']['dataset']
    score_file_path = config['paths']['score_file']

    # Model related
    alpha = np.float32(config['model']['alpha'])

    # Splitting related
    split_window_size = int(config['splitting']['window_size'])
    split_window_shift = int(config['splitting']['window_shift'])
    train_test_ratio = float(config['splitting']['train_test_ratio'])

    timeStep = 1
    epsilon = np.float32(0.01)
    top_index = int(100)
    push_threshold = float(0.1)
    dim = 512
    beta = 0.0

    track_mode = 'all'
    n_jobs = os.cpu_count()
    validate_appr = False
    detect_anomaly = True
    write_snapshot_ppvs = False
    make_directed_graph = True
    make_multi_graph = True 
 
    # Set directory
    set_dir( os.path.dirname(score_file_path) )

    # CPU
    numba.set_num_threads(n_jobs)

    # Debugging
    pp = pprint.PrettyPrinter(indent=4)

    ##### Logic to aggregate the data ########
    """
    Uses lsmanip to aggregate the data into snapshots
    Creates an associated temporary file
    """
    # Load data
    link_stream = lio.from_csv(dataset, weighted=True)

    # Compute initial timestamp based on train/test split
    num_windows = (link_stream.max_time // split_window_shift) + 1
    train_windows = int(round(num_windows * train_test_ratio))
    time_init = train_windows * split_window_shift
    initSS = train_windows

    # Snapshots from data
    edgelist = []
    for e, w in link_stream.to_weightfn().items():
        u = e[0]
        v = e[1]
        t = e[2]
        t_new_min = max(
            0,
            math.floor((t - split_window_size) / split_window_shift) + 1
        )

        t_new_max = min(
            num_windows - 1,
            math.floor(t / split_window_shift)
        )

        for t_new in range(t_new_min, t_new_max + 1):
            edgelist += [(u, v, t_new) for _ in range(int(w))]

    # Resolving paths
    parts = dataset.split("/")
    attack_dataset = f"tmp_{parts[-3]}_{parts[-2]}"

    base_dir = Path(__file__).resolve().parent
    tmp_dir = base_dir / attack_dataset 

    tmp_dir.mkdir(exist_ok=True)
     
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=True, mode="w+") as tmp:
        
        # Save tmp edge file with multi edges
        with open(tmp.name, "w") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(edgelist)

        # Run the algorithm
        dataset_name = tmp.name
        out_list = test_graph_stream(  tmp.name, dataset_name, timeStep, initSS, \
                            make_directed_graph, make_multi_graph, \
                            alpha, beta, epsilon, track_mode, top_index, dim, \
                            validate_appr, push_threshold,  \
                            detect_anomaly, write_snapshot_ppvs )
    
    anomaly_scores = out_list

    with open(score_file_path, "wb") as f:
        pickle.dump(anomaly_scores, f)

    print('DynAnom scoring done.')
