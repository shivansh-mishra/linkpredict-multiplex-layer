##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    precision_score, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score

from multiplex_hoplp_algo import normalize,cn_weight,pa_weight,jc_weight,aa_weight,\
    ra_weight,local_path_weight,cc_weight, hoplp_mul, madm_mul, nsilr_mul

import time
import random
from xlwt import Workbook
import xlrd
import datetime
import os
import sys

if __name__ == '__main__':

    starttime_full = time.time()
    var_dict_main = {}


    def auprgraph (all_graphs,layers,layer_no,nodes,algo,file_name,tao,theta,iterations):
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        print("for algo - "+str(algo))
        file_write_name = './multiplex_data/result_layer/'+algo+'/' + file_name + "_"+\
                          str(layer_no)+".txt"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        multiplex_graph = nx.Graph()
        multiplex_graph.add_nodes_from(nodes)
        multiplex_graph_adj = np.zeros((len(nodes), len(nodes)))
        multiplex_graph_wt = np.zeros((len(nodes), len(nodes)))
        no_layers = len(layers)
        full_graph = nx.Graph()
        full_graph.add_nodes_from(nodes)
        for i in range(len(layers)):
            full_graph.add_edges_from(all_graphs[i].edges)
        all_edge_no = len(sorted(full_graph.edges))
        layer_edge_no = np.zeros(len(layers))
        for i in range(len(layers)):
            layer_edge_no[i] = len(sorted(all_graphs[i].edges))
            for edge in all_graphs[i].edges:
                node1 = edge[0]
                node2 = edge[1]
                node1_index = nodes.index(node1)
                node2_index = nodes.index(node2)
                multiplex_graph_adj[node1_index][node2_index] = 1
                multiplex_graph_adj[node2_index][node1_index] = 1
                multiplex_graph_wt[node1_index][node2_index] += 1 / layer_edge_no[i]
                multiplex_graph_wt[node2_index][node1_index] += 1 / layer_edge_no[i]
        multiplex_graph_wt = multiplex_graph_wt / len(layers)
        nodes_all = nodes
        score = (all_edge_no - layer_edge_no[layer_no])/all_edge_no
        starttime_aup = time.time()
        ratio = []
        aupr = []
        recall = []
        auc = []
        avg_prec = []
        acc_score = []
        bal_acc_score = []
        f1 = []
        prec = []
        adj = multiplex_graph_adj
        G = nx.Graph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
        for i in [0.5,0.6,0.7,0.8,0.9]: # range is the fraction of edge values included in the graph
            print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
            print ("For ratio : " , i-1)
            if algo in ["cn_weight","pa_weight","jc_weight","ra_weight","aa_weight",
                        "local_path_weight","cc_weight","hoplp_mul"]:
                if algo not in ["hoplp_mul"]:
                    avg_array = avg_seq_all(all_graphs[layer_no], layer_no, score,
                                            multiplex_graph_wt, nodes_all, file_name, i, algo, iterations)
                else :
                    avg_array = avg_seq_all(all_graphs[layer_no], layer_no, score,
                                             multiplex_graph_wt, nodes_all, file_name, i, algo, iterations)
            elif algo in ["madm_mul","nsilr_mul"]:
                avg_array = avg_sota(all_graphs, layer_no, no_layers, nodes_all, file_name, i, algo, iterations)
            else:
                print("unidentified algo = "+str(algo))
                sys.exit()
            aupr.append(avg_array[0])
            recall.append(avg_array[1])
            auc.append(avg_array[2])
            avg_prec.append(avg_array[3])
            acc_score.append(avg_array[4])
            bal_acc_score.append(avg_array[5])
            f1.append(avg_array[6])
            prec.append(avg_array[7])
            ratio.append(i-1)
        print("Ratio:-", ratio)
        print("AUPR:-",aupr)
        print("Recall:-",recall)
        print("AUC:-",auc)
        print("Avg Precision:-",avg_prec)
        print("Accuracy Score:-",acc_score)
        print("Balanced Accuracy Score:-", bal_acc_score)
        print("F1 Score:-", f1)
        print("Precision Score:-", prec)
        endtime_aup = time.time()
        print('That aup took {} seconds'.format(endtime_aup - starttime_aup))

        # Workbook is created
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'Ratio')
        sheet1.write(0, 1, 'AUPR')
        sheet1.write(0, 2, 'RECALL')
        sheet1.write(0, 3, 'AUC')
        sheet1.write(0, 4, 'AVG PRECISION')
        sheet1.write(0, 5, 'ACCURACY SCORE')
        sheet1.write(0, 6, 'BAL ACCURACY SCORE')
        sheet1.write(0, 7, 'F1 MEASURE')
        sheet1.write(0, 8, 'PRECISION')
        for i in range(5):
            sheet1.write(5 - i, 0, ratio[i]*-1)
            sheet1.write(5 - i, 1, aupr[i])
            sheet1.write(5 - i, 2, recall[i])
            sheet1.write(5 - i, 3, auc[i])
            sheet1.write(5 - i, 4, avg_prec[i])
            sheet1.write(5 - i, 5, acc_score[i])
            sheet1.write(5 - i, 6, bal_acc_score[i])
            sheet1.write(5 - i, 7, f1[i])
            sheet1.write(5 - i, 8, prec[i])

        wb.save('./multiplex_data/result_layer/'+algo+'/' + file_name +"_"+str(layer_no)+".xls")

        currentDT = datetime.datetime.now()
        print(str(currentDT))

        file_all = open('./multiplex_data/result_layer/current_all.txt','a')
        text_final = "full algo = "+algo+" file name = "+file_name+" tao = "+str(tao)+\
                     " theta = "+str(theta)+" layer_n = "+ str(layer_no)+" iterations = "\
                     +str(iterations)+" time = "+str((endtime_aup - starttime_aup))+\
                     " date_time = "+str(currentDT)+"\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

        return aupr,ratio,recall,auc,avg_prec,acc_score,bal_acc_score,f1,prec


    def avg_seq_all(graph_layer, layer_no, score, adj_wt, nodes_all, file_name, ratio, algo,
                    iterations, theta=0.5, tao=10, alpha1=0.3, alpha2=0.7) :
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        start_time_ratio = time.time()
        aupr = 0
        recall = 0
        auc = 0
        avg_prec = 0
        acc_score = 0
        bal_acc_score = 0
        f1 = 0
        prec = 0
        loop = int(iterations)
        ratio = round(ratio, 1)
        graph_original = graph_layer
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("iteration = "+str(single_iter)+" of total - "+str(loop)+" for layer = "+str(layer_no))
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            # making original graph adjacency matrix
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            # finding edges and nodes of original graph
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
            edges_train = np.array(edges_original, copy=True)
            np.random.shuffle(edges_train)
            edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
            # finding training set of edges according to ratio
            graph_train = nx.Graph()
            # making graph based on the training edges
            graph_train.add_nodes_from(nodes)
            graph_train.add_edges_from(edges_train)
            adj_train = nx.adjacency_matrix(graph_train).todense()
            # making test graph by removing train edges from original
            graph_test = nx.Graph()
            graph_test.add_nodes_from(nodes)
            graph_test.add_edges_from(edges_original)
            graph_test.remove_edges_from(edges_train)
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            # sending training graph for probability matrix prediction
            adj = adj_train
            G = nx.Graph(adj_train)
            for (u, v) in G.edges():
                u_index = nodes_all.index(u)
                v_index = nodes_all.index(v)
                G.edges[u, v]['weight'] = adj_wt[u_index][v_index]

            if algo == "cn_weight":
                prob_mat = cn_weight(G)
            elif algo == "jc_weight":
                prob_mat = jc_weight(G)
            elif algo == "pa_weight":
                prob_mat = pa_weight(G)
            elif algo == "aa_weight":
                prob_mat = aa_weight(G)
            elif algo == "ra_weight":
                prob_mat = ra_weight(G)
            elif algo == "local_path_weight":
                prob_mat = local_path_weight(G)
            elif algo == "cc_weight":
                prob_mat = cc_weight(G)
            elif algo == "hoplp_mul":
                prob_mat = hoplp_mul(G)
            else:
                print("unknown algo encountered")
                sys.exit()

            prob_mat = normalize(prob_mat)
            print(score)
            prob_mat = score*prob_mat
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            # making adcancecy test from testing graph
            adj_test = nx.adjacency_matrix(graph_test).todense()
            # making new arrays to pass to function
            array_true = []
            array_pred = []
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if not graph_original.has_edge(i, j):
                        array_true.append(0)
                        array_pred.append(prob_mat[i][j])
                    if graph_test.has_edge(i, j):
                        array_true.append(1)
                        array_pred.append(prob_mat[i][j])

            pred = array_pred
            adj_test = array_true

            # return precision recall pairs for particular thresholds
            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./multiplex_data/result_layer/current_all.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " layer_n = "+str(layer_no)+\
                                 " iteration = " + str(single_iter) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            recall += recall_value
            auc += auc_value
            avg_prec += avg_prec_value
            acc_score += acc_score_value
            bal_acc_score += bal_acc_score_value
            f1 += f1_value
            prec += precision_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./multiplex_data/result_layer/current_all.txt', 'a')
        text_inside = "full pool algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " layer_n = "+str(layer_no)+ " time = " \
                            + str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, recall / loop, auc / loop, avg_prec / loop, acc_score / loop, \
               bal_acc_score / loop, f1 / loop, prec / loop


    def avg_sota(all_graphs, layer_no, layers, nodes_all, file_name, ratio, algo, iterations) :
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        start_time_ratio = time.time()
        aupr = 0
        recall = 0
        auc = 0
        avg_prec = 0
        acc_score = 0
        bal_acc_score = 0
        f1 = 0
        prec = 0
        loop = int(iterations)
        ratio = round(ratio, 1)
        graph_original = all_graphs[layer_no]
        print("avg sequential sota called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("iteration = "+str(single_iter)+" of total - "+str(loop)+" for layer = "+str(layer_no))
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            # making original graph adjacency matrix
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            # finding edges and nodes of original graph
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
            edges_train = np.array(edges_original, copy=True)
            np.random.shuffle(edges_train)
            edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
            # finding training set of edges according to ratio
            graph_train = nx.Graph()
            # making graph based on the training edges
            graph_train.add_nodes_from(nodes)
            graph_train.add_edges_from(edges_train)
            adj_train = nx.adjacency_matrix(graph_train).todense()
            # making test graph by removing train edges from original
            graph_test = nx.Graph()
            graph_test.add_nodes_from(nodes)
            graph_test.add_edges_from(edges_original)
            graph_test.remove_edges_from(edges_train)
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            if algo == "madm_mul":
                prob_mat = madm_mul(graph_train, all_graphs, layer_no, layers, nodes_all)
            elif algo == "nsilr_mul":
                prob_mat = nsilr_mul(graph_train, all_graphs, layer_no, layers, nodes_all)
            else:
                print("unknown algo encountered")
                sys.exit()

            prob_mat = normalize(prob_mat)
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            # making adcancecy test from testing graph
            adj_test = nx.adjacency_matrix(graph_test).todense()
            # making new arrays to pass to function
            array_true = []
            array_pred = []
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if not graph_original.has_edge(i, j):
                        array_true.append(0)
                        array_pred.append(prob_mat[i][j])
                    if graph_test.has_edge(i, j):
                        array_true.append(1)
                        array_pred.append(prob_mat[i][j])

            pred = array_pred
            adj_test = array_true

            # return precision recall pairs for particular thresholds
            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./multiplex_data/result_layer/current_all.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " layer_n = "+str(layer_no)+\
                                 " iteration = " + str(single_iter) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            recall += recall_value
            auc += auc_value
            avg_prec += avg_prec_value
            acc_score += acc_score_value
            bal_acc_score += bal_acc_score_value
            f1 += f1_value
            prec += precision_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./multiplex_data/result_layer/current_all.txt', 'a')
        text_inside = "full pool algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " layer_n = "+str(layer_no)+ " time = " \
                            + str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, recall / loop, auc / loop, avg_prec / loop, acc_score / loop, \
               bal_acc_score / loop, f1 / loop, prec / loop


    def aupgraph_control(file_name_array, algo_array, tao, theta, iterations = 10):
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        file_write_name = './multiplex_data/result_layer/current.txt'
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        print("inside auprgraph control")
        for file_name in file_name_array:
            for algo in algo_array:
                print("reading dataset = " + str(file_name))
                file_read = './multiplex_data/final_format/' + file_name+"_multiplex.edges"
                data = open(file_read)
                edgelist = map(lambda q: list(map(float, q.split())), data.read().split("\n")[:-1])
                data.close()
                edgelist = list(edgelist)
                layers = set()
                nodes = set()
                print("file name = "+str(file_name))
                print("total no. of edges = "+str(len(edgelist)))
                for edge in edgelist:
                    layers.add(edge[0])
                    nodes.add(edge[1])
                    nodes.add(edge[2])
                layers = list(layers)
                nodes = list(nodes)
                nodes.sort()
                print("total no. of layers = "+str(len(layers)))
                print("total no. of nodes = "+str(len(nodes)))
                all_graphs = dict()
                for i in range(len(layers)):
                    temp = nx.Graph()
                    temp.add_nodes_from(nodes)
                    edgelist_curr = list()
                    for edge in edgelist:
                        if edge[0] == layers[i]:
                            edgelist_curr.append([edge[1],edge[2]])
                    temp.add_edges_from(edgelist_curr)
                    temp.remove_edges_from(nx.selfloop_edges(temp))
                    adj = nx.adjacency_matrix(temp).todense()
                    temp = nx.Graph(adj)
                    all_graphs[i] = temp
                nodes = list(range(len(nodes)))
                for i in range(len(layers)):
                    print("layer = "+str(i))
                    print("nodes = "+str(len(all_graphs[i])))
                    print("edges = "+str(len(sorted(all_graphs[i].edges))))
                    auprgraph(all_graphs,layers,i,nodes,algo,file_name,tao,theta,iterations)


    def result_parser_combine(file_name_array,algo_all):
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        for file_name in file_name_array:
            print("reading dataset = " + str(file_name))
            file_read = './multiplex_data/final_format/' + file_name + "_multiplex.edges"
            data = open(file_read)
            edgelist = map(lambda q: list(map(float, q.split())), data.read().split("\n")[:-1])
            data.close()
            edgelist = list(edgelist)
            layers = set()
            nodes = set()
            print("file name = " + str(file_name))
            print("total no. of edges = " + str(len(edgelist)))
            for edge in edgelist:
                layers.add(edge[0])
                nodes.add(edge[1])
                nodes.add(edge[2])
            layers = list(layers)
            nodes = list(nodes)
            print("total no. of layers = " + str(len(layers)))
            print("total no. of nodes = " + str(len(nodes)))
            for layer_no in range(len(layers)):
                file_write_name = './multiplex_data/result_layer/' + file_name + "_"+\
                                  str(layer_no)+"_combine.xls"
                os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
                # Workbook is created
                wb_write = Workbook()
                # add_sheet is used to create sheet.
                AUPR = wb_write.add_sheet('AUPR', cell_overwrite_ok=True)
                RECALL = wb_write.add_sheet('RECALL', cell_overwrite_ok=True)
                AUC = wb_write.add_sheet('AUC', cell_overwrite_ok=True)
                AVG_PREC = wb_write.add_sheet('AVG PREC', cell_overwrite_ok=True)
                ACC_SCORE = wb_write.add_sheet('ACC SCORE', cell_overwrite_ok=True)
                BAL_ACC_SCORE = wb_write.add_sheet('BAL ACC SCORE', cell_overwrite_ok=True)
                F1_SCORE = wb_write.add_sheet('F1 SCORE', cell_overwrite_ok=True)
                PRECISION = wb_write.add_sheet('PRECISION', cell_overwrite_ok=True)
                sheet_array = [AUPR, RECALL, AUC, AVG_PREC, ACC_SCORE, BAL_ACC_SCORE, F1_SCORE, PRECISION]
                for sheet_single in sheet_array:
                    sheet_single.write(0, 0, 'Ratio')
                    sheet_single.write(1, 0, '0.1')
                    sheet_single.write(2, 0, '0.2')
                    sheet_single.write(3, 0, '0.3')
                    sheet_single.write(4, 0, '0.4')
                    sheet_single.write(5, 0, '0.5')
                current_algo = 1
                for algo in algo_all:
                    single_algo_file = "./multiplex_data/result_layer/" + str(algo) + '/' + \
                                       file_name + "_"+str(layer_no)+".xls"
                    wb_read = xlrd.open_workbook(single_algo_file)
                    main_sheet = wb_read.sheet_by_name('Sheet 1')
                    for sheet_single in sheet_array:
                        sheet_single.write(0, current_algo, str(algo).upper())
                    for row_read in range(5):
                        row_read += 1
                        row_write = row_read
                        for col_read in range(8):
                            col_read += 1
                            print("reading--" + file_name + " --of algo--" + algo)
                            value = float(main_sheet.cell(row_read, col_read).value)
                            value = round(value, 5)
                            sheet_no = col_read - 1
                            sheet_array[sheet_no].write(row_write, current_algo, value)
                    current_algo = current_algo + 1
                wb_write.save(file_write_name)

        wb_dataset_write = Workbook()
        file_dataset_write_name = './multiplex_data/result_layer/all_datasets_combine_layer.xls'
        sheet_name_array = ['AUPR', 'RECALL', 'AUC', 'AVG PREC', 'ACC SCORE', 'BAL ACC SCORE',
                            'F1 SCORE', 'PRECISION']
        AUPR_write = wb_dataset_write.add_sheet(sheet_name_array[0], cell_overwrite_ok=True)
        RECALL_write = wb_dataset_write.add_sheet(sheet_name_array[1], cell_overwrite_ok=True)
        AUC_write = wb_dataset_write.add_sheet(sheet_name_array[2], cell_overwrite_ok=True)
        AVG_PREC_write = wb_dataset_write.add_sheet(sheet_name_array[3], cell_overwrite_ok=True)
        ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[4], cell_overwrite_ok=True)
        BAL_ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[5], cell_overwrite_ok=True)
        F1_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[6], cell_overwrite_ok=True)
        PRECISION_write = wb_dataset_write.add_sheet(sheet_name_array[7], cell_overwrite_ok=True)
        sheet_dataset_write_array = [AUPR_write, RECALL_write, AUC_write, AVG_PREC_write, ACC_SCORE_write,
                                     BAL_ACC_SCORE_write, F1_SCORE_write, PRECISION_write]
        count = 0
        for file_name in file_name_array:
            print("reading dataset = " + str(file_name))
            file_read = './multiplex_data/final_format/' + file_name + "_multiplex.edges"
            data = open(file_read)
            edgelist = map(lambda q: list(map(float, q.split())), data.read().split("\n")[:-1])
            data.close()
            edgelist = list(edgelist)
            layers = set()
            nodes = set()
            print("file name = " + str(file_name))
            print("total no. of edges = " + str(len(edgelist)))
            for edge in edgelist:
                layers.add(edge[0])
                nodes.add(edge[1])
                nodes.add(edge[2])
            layers = list(layers)
            nodes = list(nodes)
            print("total no. of layers = " + str(len(layers)))
            print("total no. of nodes = " + str(len(nodes)))
            for layer_no in range(len(layers)):
                file_read_name = './multiplex_data/result_layer/' + file_name +"_"+str(layer_no) +"_combine.xls"
                wb_read = xlrd.open_workbook(file_read_name)
                AUPR = wb_read.sheet_by_name(sheet_name_array[0])
                RECALL = wb_read.sheet_by_name(sheet_name_array[1])
                AUC = wb_read.sheet_by_name(sheet_name_array[2])
                AVG_PREC = wb_read.sheet_by_name(sheet_name_array[3])
                ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[4])
                BAL_ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[5])
                F1_SCORE = wb_read.sheet_by_name(sheet_name_array[6])
                PRECISION_SCORE = wb_read.sheet_by_name(sheet_name_array[7])
                sheet_read_array = [AUPR, RECALL, AUC, AVG_PREC, ACC_SCORE, BAL_ACC_SCORE, F1_SCORE, PRECISION_SCORE]
                write_row = file_name_array.index(file_name) + 1
                for sheet_no in range(len(sheet_read_array)):
                    sheet_dataset_write_array[sheet_no].write(0, 1, 'Ratio')
                    sheet_dataset_write_array[sheet_no].write(1 + count * 6, 1, '0.1')
                    sheet_dataset_write_array[sheet_no].write(2 + count * 6, 1, '0.2')
                    sheet_dataset_write_array[sheet_no].write(3 + count * 6, 1, '0.3')
                    sheet_dataset_write_array[sheet_no].write(4 + count * 6, 1, '0.4')
                    sheet_dataset_write_array[sheet_no].write(5 + count * 6, 1, '0.5')
                    sheet_dataset_write_array[sheet_no].write(0, 0, 'FILE_NAME')
                    sheet_dataset_write_array[sheet_no].write(1 + count * 6, 0, str(file_name))
                    for ratio in range(5):
                        read_row = ratio + 1
                        write_row = ratio + 1 + count * 6
                        for algo_no in range(len(algo_all)):
                            read_col = algo_no + 1
                            write_col = algo_no + 2
                            value = sheet_read_array[sheet_no].cell(read_row, read_col).value
                            print("value read = " + str(value))
                            # sheet_dataset_write_array[sheet_no].write(count * 6, write_col, str(algo_all[algo_no]).upper())
                            sheet_dataset_write_array[sheet_no].write(0, write_col, str(algo_all[algo_no]).upper())
                            sheet_dataset_write_array[sheet_no].write(write_row, write_col, value)
                count += 1
        wb_dataset_write.save(file_dataset_write_name)


    def dataset_info(file_name_array):
        ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

        file_write_name = "./multiplex_data/layer_info/layer_info.xls"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        print("inside dataset info control")
        row_no = 0
        wb_write = Workbook()
        # add_sheet is used to create sheet.
        sheet_write = wb_write.add_sheet('Sheet 1', cell_overwrite_ok=True)
        sheet_write.write(row_no, 0, 'DATASET')
        sheet_write.write(row_no, 1, 'LAYER')
        sheet_write.write(row_no, 2, 'NODES')
        sheet_write.write(row_no, 3, 'EDGES')
        sheet_write.write(row_no, 4, 'AVG SHORTP.')
        sheet_write.write(row_no, 5, 'CLUSTER COE.')
        sheet_write.write(row_no, 6, 'ASSOR. COE.')
        sheet_write.write(row_no, 7, 'AVG CONNECT.')
        sheet_write.write(row_no, 8, 'DENSITY')
        sheet_write.write(row_no, 9, 'TRANSITIVITY')
        sheet_write.write(row_no, 10, 'HETEROGENITY')
        for file_name in file_name_array:
            row_no = row_no + 1
            sheet_write.write(row_no, 0, str(file_name))
            print("reading dataset = " + str(file_name))
            file_read = './multiplex_data/final_format/' + file_name + "_multiplex.edges"
            data = open(file_read)
            edgelist = map(lambda q: list(map(float, q.split())), data.read().split("\n")[:-1])
            data.close()
            edgelist = list(edgelist)
            layers = set()
            nodes = set()
            print("file name = " + str(file_name))
            print("total no. of edges = " + str(len(edgelist)))
            layer_dict = {}
            for edge in edgelist:
                layers.add(edge[0])
                layer_no = edge[0]
                temp = set()
                if layer_no in layer_dict.keys():
                    temp = layer_dict[layer_no]
                nodes.add(edge[1])
                nodes.add(edge[2])
                temp.add(edge[1])
                temp.add(edge[2])
                layer_dict[layer_no] = temp
            layers = list(layers)
            nodes = list(nodes)
            print("total no. of layers = " + str(len(layers)))
            print("total no. of nodes = " + str(len(nodes)))
            all_graphs = dict()
            for i in range(len(layers)):
                temp = nx.Graph()
                temp.add_nodes_from(nodes)
                edgelist_curr = list()
                for edge in edgelist:
                    if edge[0] == layers[i]:
                        edgelist_curr.append([edge[1], edge[2]])
                temp.add_edges_from(edgelist_curr)
                temp.remove_edges_from(nx.selfloop_edges(temp))
                all_graphs[i] = temp
            for i in range(len(layers)):
                print("layer = " + str(i))
                sheet_write.write(row_no, 1, str(i + 1))
                print("nodes = " + str(len(all_graphs[i])))
                sheet_write.write(row_no, 2, str(len(layer_dict[i + 1])))
                print("edges = " + str(len(sorted(all_graphs[i].edges))))
                sheet_write.write(row_no, 3, str(len(sorted(all_graphs[i].edges))))
                pathlengths = []
                for v in all_graphs[i].nodes():
                    spl = dict(nx.single_source_shortest_path_length(all_graphs[i], v))
                    for p in spl:
                        pathlengths.append(spl[p])
                avg_short_path = sum(pathlengths) / len(pathlengths)
                sheet_write.write(row_no, 4, str(round(avg_short_path, 2)))
                avg_cluster = nx.average_clustering(all_graphs[i])
                sheet_write.write(row_no, 5, str(round(avg_cluster, 2)))
                degre_assor = nx.degree_assortativity_coefficient(all_graphs[i])
                sheet_write.write(row_no, 6, str(round(degre_assor, 2)))
                # avg_node_conn = nx.algorithms.connectivity.average_node_connectivity(all_graphs[i])
                # sheet_write.write(row_no, 7, str(round(avg_node_conn, 2)))
                sheet_write.write(row_no, 8, str(round(nx.density(all_graphs[i]), 2)))
                sheet_write.write(row_no, 9, str(round(nx.transitivity(all_graphs[i]), 2)))
                numerator = 0
                denominator = 0
                print("calculating heterogeneity")
                heterogenity = 0
                '''node_no = len(all_graphs[i])
                for node in all_graphs[i]:
                    node_neighbor_no = 0
                    for node_neighbor in all_graphs[i].neighbors(node):
                        node_neighbor_no += 1
                    numerator += (node_neighbor_no ** 2)
                    denominator += node_neighbor_no
                numerator /= node_no
                denominator /= node_no
                denominator = denominator ** 2
                heterogenity = numerator / denominator'''
                print("writing after skip layer")
                sheet_write.write(row_no, 10, str(round(heterogenity, 2)))
                row_no = row_no + 1
            print("creating multiplex graph")
            multiplex_graph = nx.Graph()
            multiplex_graph.add_nodes_from(nodes)
            multiplex_graph_adj = np.zeros((len(nodes), len(nodes)))
            multiplex_graph_wt = np.zeros((len(nodes), len(nodes)))
            for node1 in nodes:
                for node2 in nodes:
                    node1_index = nodes.index(node1)
                    node2_index = nodes.index(node2)
                    temp = 0
                    for i in range(len(layers)):
                        if all_graphs[i].has_edge(node1, node2):
                            temp += 1
                            multiplex_graph_adj[node1_index][node2_index] = 1
                    multiplex_graph_wt[node1_index][node2_index] = temp / len(layers)
            adj = multiplex_graph_wt
            G = nx.Graph(adj)
            G.remove_edges_from(nx.selfloop_edges(G))
            sheet_write.write(row_no, 1, str("COMBINE"))
            print("nodes = " + str(len(G)))
            sheet_write.write(row_no, 2, str(len(G)))
            print("edges = " + str(len(sorted(G.edges))))
            sheet_write.write(row_no, 3, str(len(sorted(G.edges))))
            pathlengths = []
            for v in G:
                spl = dict(nx.single_source_shortest_path_length(G, v))
                for p in spl:
                    pathlengths.append(spl[p])
            avg_short_path = sum(pathlengths) / len(pathlengths)
            sheet_write.write(row_no, 4, str(round(avg_short_path, 2)))
            avg_cluster = nx.average_clustering(G)
            sheet_write.write(row_no, 5, str(round(avg_cluster, 2)))
            degre_assor = nx.degree_assortativity_coefficient(G)
            sheet_write.write(row_no, 6, str(round(degre_assor, 2)))
            # avg_node_conn = nx.algorithms.connectivity.average_node_connectivity(all_graphs[i])
            # sheet_write.write(row_no, 7, str(round(avg_node_conn, 2)))
            sheet_write.write(row_no, 8, str(round(nx.density(G), 2)))
            sheet_write.write(row_no, 9, str(round(nx.transitivity(G), 2)))
            numerator = 0
            denominator = 0
            print("calculating heterogeneity")
            heterogenity = 0
            node_no = len(G)
            '''for node in G:
                node_neighbor_no = len(sorted(G.neighbors(node)))
                numerator += (node_neighbor_no ** 2)
                denominator += node_neighbor_no
            numerator /= node_no
            denominator /= node_no
            denominator = denominator ** 2
            heterogenity = numerator / denominator'''
            print("writing after skip combine")
            sheet_write.write(row_no, 10, str(round(heterogenity, 2)))
            row_no = row_no + 1

        wb_write.save(file_write_name)


    algo_array_sota = ["cn_weight","pa_weight","jc_weight","aa_weight","ra_weight",
                       "cc_weight","local_path_weight","nsilr_mul","madm_mul",
                       "hoplp_mul"]
    iterations = 100
    file_name_array = ['Vickers-Chan-7thGraders', 'Kapferer-Tailor-Shop',
                        'CKM-Physicians-Innovation']
    dataset_info(file_name_array)
    aupgraph_control(file_name_array,algo_array_sota,15,0.5,iterations)
    result_parser_combine(file_name_array,algo_array_sota)
