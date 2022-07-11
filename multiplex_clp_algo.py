import networkx as nx
import numpy as np
import math
import sys
import datetime
import random


def normalize(n):
    max = 0
    max_n = 0
    for i in range(len(n)):
        for j in range(len(n[0])):
            if n[i][j] >= 0:
                if max < n[i][j]: max = n[i][j]
            else:
                if max_n > n[i][j]: max_n = n[i][j]
    if max < max_n * -1: max = max_n * -1
    for i in n:
        if max != 0:
            for j in range(len(i)):
                i[j] = i[j] / max
    return n


def cn_weight(graph):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj),len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2 :
                neighbors_all = nx.common_neighbors(graph,node1,node2)
                for single in neighbors_all:
                    common[node1][node2] += graph.edges[node1,single]['weight'] + graph.edges[node2,single]['weight']
            else :
                common[node1][node2] = common[node2][node1]
    return common


def jc_weight(graph):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj), len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2:
                denom = 0
                neighbors_node1 = nx.neighbors(graph,node1)
                neighbors_node2 = nx.neighbors(graph,node2)
                for single in neighbors_node1:
                    denom += graph.edges[node1, single]['weight']
                for single in neighbors_node2:
                    denom += graph.edges[node2, single]['weight']
                if denom != 0:
                    neighbors_all = nx.common_neighbors(graph, node1, node2)
                    for single in neighbors_all:
                        common[node1][node2] += graph.edges[node1, single]['weight'] + graph.edges[node2, single]['weight']
                    common[node1][node2] = common[node1][node2]/denom
            else:
                common[node1][node2] = common[node2][node1]
    return common


def pa_weight(graph):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj),len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2 :
                neighbors_node1 = nx.neighbors(graph,node1)
                neighbors_node2 = nx.neighbors(graph,node2)
                sum_node1 = 0
                sum_node2 = 0
                for single in neighbors_node1:
                    sum_node1 += graph.edges[node1,single]['weight']
                for single in neighbors_node2:
                    sum_node2 += graph.edges[node2,single]['weight']
                common[node1][node2] = sum_node1 * sum_node2
            else :
                common[node1][node2] = common[node2][node1]
    return common


def aa_weight(graph):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj),len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2 :
                neighbors_all = nx.common_neighbors(graph,node1,node2)
                for single in neighbors_all:
                    num = graph.edges[node1,single]['weight'] + graph.edges[node2,single]['weight']
                    denom = 1
                    neighbors_single = nx.neighbors(graph,single)
                    for single_inside in neighbors_single:
                        denom += graph.edges[single_inside,single]['weight']
                    denom = math.log(denom)
                    if denom != 0:
                        common[node1][node2] += num/denom
            else :
                common[node1][node2] = common[node2][node1]
    return common


def ra_weight(graph):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj),len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2 :
                neighbors_all = nx.common_neighbors(graph,node1,node2)
                for single in neighbors_all:
                    num = graph.edges[node1,single]['weight'] + graph.edges[node2,single]['weight']
                    denom = 0
                    neighbors_single = nx.neighbors(graph,single)
                    for single_inside in neighbors_single:
                        denom += graph.edges[single_inside,single]['weight']
                    if denom != 0:
                        common[node1][node2] += num/denom
            else :
                common[node1][node2] = common[node2][node1]
    return common


def local_path_weight(graph,parameter=0.05):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros((len(adj), len(adj)))
    for node1 in graph:
        for node2 in graph:
            if node1 > node2:
                common_n = nx.common_neighbors(graph,node1,node2)
                for single in common_n:
                    common[node1][node2] += graph.edges[node1,single]['weight'] + graph.edges[node2,single]['weight']
                neighbors_node1 = nx.neighbors(graph,node1)
                neighbors_node2 = nx.neighbors(graph,node2)
                for single_node1 in neighbors_node1:
                    for single_node2 in neighbors_node2:
                        if graph.has_edge(single_node1,single_node2):
                            common[node1][node2] += parameter*(
                                    graph.edges[node1, single_node1]['weight'] +
                                    graph.edges[node2, single_node2]['weight'] +
                                    graph.edges[single_node1, single_node2]['weight'])
    return common


def cc_weight (graph) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    adj = nx.adjacency_matrix(graph).todense()
    triangles = nx.triangles(graph)
    common = np.zeros((len(adj), len(adj)))
    cc = np.zeros(len(adj))
    for node in graph:
        if graph.degree(node) > 1:
            cc[node] = triangles[node] / (graph.degree(node) * (graph.degree(node) - 1) / 2)
            neighbors1 = nx.neighbors(graph,node)
            neighbors2 = nx.neighbors(graph,node)
            element = 0
            for single1 in neighbors1:
                for single2 in neighbors2:
                    if graph.has_edge(single1,single2):
                        element += graph.edges[node,single1]['weight'] + graph.edges[node,single2]['weight']
            neighbors = nx.neighbors(graph, node)
            denom = 0
            for single in neighbors:
                denom += graph.edges[node,single]['weight']
            if denom > 0 :
                denom = denom * 2 / graph.degree(node)
                element = element /denom
                cc[node] = cc[node]*element
            else :
                cc[node] = 0
    for i in range(len(adj)):
        for j in range(len(adj)):
            common[i][j] = cc[i]+cc[j]

    return common


def max_comm_label(node):
    global var_dict
    G = var_dict['graph']
    all_labels = set()
    # print("initially for node "+str(node)+" label is "+str(var_dict[node]))
    for node_neighbour in G.neighbors(node):
        all_labels.add(var_dict[node_neighbour])
    prob_actual = 1
    label_actual = var_dict[node]
    for label in all_labels:
        # print("for label "+str(label))
        prob_new = 1
        for node_chk in G.neighbors(node):
            # print("u is-"+str(u)+" v is-"+str(v))
            if var_dict[node_chk] == label:
                # print("prob_new = "+str(prob_new)+" edge weight "+str(G[node][node_chk]['weight']))
                chk = 0
                if G.has_edge(node, node_chk):
                    chk = G[node][node_chk]['weight'] * var_dict['node_wt'][node_chk]
                if var_dict['influence'][node][node_chk] == 1:
                    # print("influence and edge weight true for "+str(node)+"-"+str(node_chk))
                    prob_new = prob_new * (1 - chk)
        if prob_new < prob_actual:
            prob_actual = prob_new
            label_actual = label
            var_dict[node] = label
    # print("after max_comm_label for node " + str(node) + " label is " + str(var_dict[node]))
    return label_actual


def detachability(label):
    global var_dict
    G = var_dict['graph']
    internal = 0
    external = 0
    DZ = 0
    # node and node neighbour only taken into account
    for node in G:
        if var_dict[node] == label:
            for node_neighbour in G.neighbors(node):
                if var_dict[node_neighbour] == label:
                    internal = internal + G[node][node_neighbour]['weight']*var_dict['node_wt'][node_neighbour]
                else:
                    external = external + G[node][node_neighbour]['weight']*var_dict['node_wt'][node_neighbour]
    if internal + external != 0:
        DZ = internal / (internal + external)
    return DZ


def clp_id_cluster_multiplex(graph, tao=15, theta=0.5, feature_no=0, alpha=0.8):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks
    print("inside clustering for clp multiplex")
    global var_dict
    global nodes_per_label
    nodes_per_label = {}
    var_dict = {}
    # adj = nx.adjacency_matrix(graph).todense()
    adj = nx.to_numpy_array(graph)
    G = graph.copy()
    var_dict['graph'] = G
    prev = np.zeros((len(adj), len(adj)))
    print("No. of edges - " + str(len(G.edges())))
    i = 1
    # giving node labels their number by default
    A = np.zeros((len(adj), len(adj)))
    var_dict['influence'] = A
    # making default value of A as -1, non edge
    for i in range(len(adj)):
        for j in range(len(adj)):
            A[i][j] = -1
    for node in G:
        var_dict[node] = node
        for node_neighbour in G.neighbors(node):
            if G.edges[node, node_neighbour]['weight'] > random.uniform(0, 1):
                A[node][node_neighbour] = 1
            else:
                A[node][node_neighbour] = 0
    node_wt = np.zeros(len(adj))
    var_dict['node_wt'] = node_wt
    # making default value of A as -1, non edge
    for i in range(len(adj)):
        node_wt[i] = random.uniform(0, 1)
    # checking for default labels
    # for node in G :
    # print(str(node)+" has default label "+str(var_dict[node]))
    i = 0
    while i <= tao:
        print("for i = " + str(i))
        for node in G:
            # print("for node = "+str(node))
            old_label = var_dict[node]
            new_label = max_comm_label(node)
            var_dict[node] = new_label
            # if old_label != new_label :
            # print("node = "+str(node)+" has new label "+str(var_dict[node])+" old label was "+str(old_label))
        i = i + 1
        total_labels = set()
        for node in G:
            total_labels.add(var_dict[node])
            # print(str(node)+"has final label"+str(var_dict[node]))
        print("number of labels left partition " + str(len(total_labels)))
        # for label in total_labels:
        # print("final labels " + str(label))
    all_labels = set()
    for node in G:
        all_labels.add(var_dict[node])
    print("labels before dissolution = "+str(len(all_labels)))
    for label in all_labels:
        DZ1 = detachability(label)
        # print("label-"+str(label)+" has detachability "+str(DZ1))
        if DZ1 < theta:
            # print("inside detachability less than threshold")
            just_neighbour = set()
            outer = set()
            TW = np.zeros(len(adj))
            for node in G:
                if var_dict[node] == label:
                    for node_neighbour in G.neighbors(node):
                        just_neighbour.add(node_neighbour)
                        if var_dict[node_neighbour] != label:
                            TW[node_neighbour] += 1
                else:
                    outer.add(node)
            NE = just_neighbour.intersection(outer)
            c_max = 0
            for node_inter in NE:
                if TW[node_inter] > c_max:
                    c_max = TW[node_inter]
                    # print("for label "+str(label)+" c max is "+str(c_max))
            NS = set()
            for node_inter in NE:
                if TW[node_inter] == c_max:
                    NS.add(node_inter)
            CS_label = set()
            for node in NS:
                CS_label.add(var_dict[node])
            MID = -99999
            new_label = label
            for label_other in CS_label:
                factor2 = detachability(label_other)
                # to change label_other to label find all node with label store and change
                to_be_changed = set()
                for node in G:
                    if var_dict[node] == label_other:
                        to_be_changed.add(node)
                        var_dict[node] = label
                factor1 = detachability(label)
                # change back to label_other
                for node in to_be_changed:
                    var_dict[node] = label_other
                TID = factor1 - factor2
                if TID > MID:
                    # print("label change criteria met")
                    MID = TID
                    new_label = label_other
            for node in G:
                if var_dict[node] == label:
                    # if label != new_label :
                    # print("changed label of node "+str(node))
                    var_dict[node] = new_label

    total_labels = set()
    for node in G:
        total_labels.add(var_dict[node])
        # print(str(node)+"has final label"+str(var_dict[node]))
    print("number of labels left finally after dissolution " + str(len(total_labels)))
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    for label in total_labels:
        # print("final labels "+str(label))
        count = 0
        nodes_per_label[label] = count
        max_dist = 0
        for node in G:
            if var_dict[node] == label:
                count += 1
        '''for node1 in G:
            for node2 in G:
                if node1 < node2:
                    if var_dict[node1] == var_dict[node2] and node1 != node2:
                        try:
                            curr_max = nx.shortest_path_length(G, node1, node2, weight='weight')
                        except:
                            curr_max = 0
                        if curr_max > max_dist: max_dist = curr_max
        cluster_distance[label] = max_dist
        if max_dist == 0:
            print("0 inter cluster distance")
            sys.exit()
        if max_dist > len(G):
            print("too high distance")
            sys.exit()'''
        nodes_per_label[label] = count
        # print("for label "+str(label)+" number of nodes "+str(nodes_per_label[label])+" with count "+str(count))
    #print("cluster_distance len = " + str(len(cluster_distance)))
    #print("cluster_distance = " + str(cluster_distance))
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    print("before close")
    # sys.exit()
    cluster_matrix = np.zeros((len(adj), len(adj)))
    close = np.zeros((len(adj), len(adj)))
    ds_all = 0
    for node1 in G:
        if node1 %100 == 0:
            print(str(node1)+" out of "+str(len(adj))+" for cluster matrix")
            currentDT = datetime.datetime.now()
            print(str(currentDT))
        for node2 in G:
            if node1 < node2:
                '''try:
                    ds = nx.shortest_path_length(G,node1,node2)
                except:
                    ds = 9999999999'''
                ds = 9999999999
                #print("trying to find ds -"+str((node1,node2)))
                if G.has_edge(node1,node2): ds = 1
                else:
                    common_n = nx.common_neighbors(G,node1,node2)
                    cn_len = len(sorted(common_n))
                    if cn_len > 0: ds = 2
                    else:
                        neighhbors1 = G.neighbors(node1)
                        check = 0
                        for single1 in neighhbors1:
                            for single2 in nx.common_neighbors(G, single1, node2):
                                check = 1
                        if check == 1: ds = 3
                if ds <= 1 : close[node1][node2] = 1
                elif ds == 2 :
                    common_n = nx.common_neighbors(G,node1,node2)
                    count = 0
                    for single in common_n:
                        count += 1
                        close[node1][node2] = (G.edges[node1, single]['weight']*node_wt[single]+
                                               G.edges[node2, single]['weight']*node_wt[node2])*math.exp(-1)
                        break
                    #close[node1][node2] = close[node1][node2]/count
                elif ds == 3 :
                    neighhbors1 = G.neighbors(node1)
                    check = 0
                    count = 0
                    for single1 in neighhbors1:
                        for single2 in nx.common_neighbors(G,single1,node2):
                            if G.has_edge(node1,single1) and G.has_edge(node2,single2) and G.has_edge(single1,single2):
                                check = 1
                                count += 1
                                close[node1][node2] = (G.edges[node1, single1]['weight'] * node_wt[single1] +
                                                       G.edges[single2, single1]['weight'] * node_wt[single2] +
                                                       G.edges[node2, single2]['weight'] * node_wt[node2]) * math.exp(-2)
                                break
                        if check == 1 : break
                    #close[node1][node2] = close[node1][node2]/count
                else: close[node1][node2] = 0
            else:
                close[node1][node2] = close[node2][node1]
    currentDT = datetime.datetime.now()
    print("after close - " + str(currentDT))
    chk_close = close.sum()
    if chk_close == 0:
        print("problem in summation "+str(chk_close))
        sys.exit()
    similarity_index = np.copy(close)
    cluster_index = np.zeros((len(adj),len(adj)))
    for node1 in G:
        for node2 in G:
            if var_dict[node1] == var_dict[node2]:
                cluster_index[node1][node2] = alpha
    common = np.zeros((len(adj),len(adj)))
    triangles = nx.triangles(G)
    for node1 in G:
        if node1 %100 == 0:
            print(str(node1)+" out of "+str(len(adj))+" final calculation")
            currentDT = datetime.datetime.now()
            print(str(currentDT))
        for node2 in G:
            if node1 < node2:
                set_feature = set()
                if feature_no == 0:
                    for single in nx.common_neighbors(G, node1, node2):
                        set_feature.add(single)
                if feature_no == 1:
                    for neighbor in G.neighbors(node1): set_feature.add(neighbor)
                    for neighbor in G.neighbors(node2): set_feature.add(neighbor)
                if feature_no == 2:
                    for level1 in nx.common_neighbors(G, node1, node2):
                        for level2 in nx.common_neighbors(G, node1, node2):
                            if level2 != level1 and G.has_edge(level1,level2):
                                set_feature.add(level2)
                                set_feature.add(level1)
                if feature_no == 3:
                    for level1 in nx.common_neighbors(G, node1, node2):
                        if triangles[level1] != 0:
                            for triangle_node1 in G.neighbors(level1):
                                for triangle_node2 in G.neighbors(level1):
                                    if G.has_edge(triangle_node1, triangle_node2):
                                        set_feature.add(triangle_node1)
                                        set_feature.add(triangle_node2)
                                        set_feature.add(level1)
                if feature_no == 4:
                    for level1 in G.neighbors(node1):
                        set_feature.add(level1)
                        for level2 in G.neighbors(level1):
                            set_feature.add(level2)
                    for level1 in G.neighbors(node2):
                        set_feature.add(level1)
                        for level2 in G.neighbors(level1):
                            set_feature.add(level2)
                    if node1 in set_feature: set_feature.remove(node1)
                    if node2 in set_feature: set_feature.remove(node2)
                numerator = 0
                denominator = 0
                if set_feature != set():
                    for single in set_feature:
                        numerator += similarity_index[node1][single]*cluster_index[node1][single] + \
                                    similarity_index[single][node2]*cluster_index[single][node2]
                        denominator = 0
                        for neighbor in G.neighbors(single):
                            denominator += similarity_index[single][neighbor]*cluster_index[single][neighbor]
                        # denominator = denominator**2
                if denominator!= 0: common[node1][node2] += numerator / denominator
            else:
                common[node1][node2] = common[node2][node1]
    link_pred = np.zeros((len(adj), len(adj)))
    cluster_matrix = common
    link_pred = common
    return link_pred



def madm_mul (graph, all_graphs, layer_no, layers, nodes_all) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    layers = int(layers)
    all_graphs[layer_no] = graph
    print("running madm mul sota for layer - "+str(layer_no)+
          " out of - "+str(layers))
    G = graph
    adj = nx.adjacency_matrix(graph).todense()
    r_layer = np.zeros(shape=(layers, layers))
    print("computing layer similarity")
    for i in range(layers):
        for j in range(layers):
            if i < j:
                num = 0
                denom1 = 0
                denom2 = 0
                for node1 in range(len(nodes_all)):
                    for node2 in range(len(nodes_all)):
                        if node1 < node2:
                            if all_graphs[i].has_edge(node1,node2) and all_graphs[j].has_edge(node1,node2):
                                num += 1
                            if all_graphs[i].has_edge(node1, node2): denom1 += 1
                            if all_graphs[j].has_edge(node1,node2): denom2 += 1
                r_layer[i][j] = num / (math.sqrt(denom1)*math.sqrt(denom2))
            else:
                r_layer[i][j] = r_layer[j][i]
    T_alpha = 1
    for i in range(layers):
        if i != layer_no:
            T_alpha += r_layer[layer_no][i]
    weight_alpha = 1 / T_alpha
    weight_beta = np.zeros(layers)
    for i in range(layers):
        if i != layer_no:
            weight_beta[i] = r_layer[layer_no][i]/T_alpha
    print("curent layer weight - "+str(weight_alpha))
    print("layer weight array - "+str(weight_beta))
    m = len(nodes_all) ** 2 - len(graph)
    print("possible edges = "+str(m))
    Y = np.zeros(shape = (m, layers))
    Y_new = list()
    node_pair = list()
    count = -1
    max = -1
    print("before Y matrix")
    for node1 in range(len(nodes_all)):
        for node2 in range(len(nodes_all)):
            if node1 != node2 and not G.has_edge(node1,node2):
                curr_list = list()
                count += 1
                col = 0
                Y[count][col] = len(sorted(nx.common_neighbors(graph, node1, node2)))
                preds = nx.jaccard_coefficient(G, [(node1, node2)])
                for u, v, p in preds: Y[count][col] = p
                curr_list.append(Y[count][col])
                if max < Y[count][col]:
                    max = Y[count][col]
                for i in range(layers):
                    if i != layer_no:
                        col += 1
                        if all_graphs[i].has_edge(node1,node2):
                            Y[count][col] = 1
                        else:
                            Y[count][col] = 0
                        curr_list.append(Y[count][col])
                node_pair.append([node1,node2])
                Y_new.append(curr_list)
    print("before Y normalization")
    Y = Y_new
    m = len(Y)
    print("new m = "+str(m))
    if max != -1:
        for i in range(m):
            Y[i][0] = Y[i][0]/max
    f_plus = np.zeros(layers)
    f_minus = np.zeros(layers)
    print("before f plus and f minus")
    for j in range(layers):
        max = -1
        min = 99999999
        for i in range(m):
            if Y[i][j] > max: max = Y[i][j]
            if Y[i][j] < min: min = Y[i][j]
        f_plus[j] = max
        f_minus[j] = min
    c_plus = np.zeros(shape = (m, layers))
    c_minus = np.zeros(shape = (m, layers))
    ideality_score = np.zeros(shape=(m, layers))
    final_score = np.zeros(m)
    for i in range(m):
        for j in range(layers):
            plus = -2 * ((Y[i][j] - f_plus[j])**2) / ((f_plus[j]-f_minus[j])**2)
            minus = -2 * ((Y[i][j] - f_minus[j])**2) / ((f_plus[j]-f_minus[j])**2)
            c_plus[i][j] = math.exp(plus)
            c_minus[i][j] = math.exp(minus)
            ideality_score[i][j] = c_plus[i][j] / (c_plus[i][j]+c_minus[i][j])
        for j in range(layers):
            if j != layer_no:
                final_score[i] += weight_beta[j] * ideality_score[i][j]
            else:
                final_score[i] += weight_alpha * ideality_score[i][j]
    common = np.zeros(shape = (len(nodes_all), len(nodes_all)))
    diff_value = set()
    for i in range(m):
        node1 = int(node_pair[i][0])
        node2 = int(node_pair[i][1])
        if node1 != node2:
            common[node1][node2] = final_score[i]
            diff_value.add(final_score[i])
            #print(str(node1) + " - " + str(node2) + " - " + str(final_score[i]))
    if len(diff_value) < 5:
        print("not much difference")
        #sys.exit()
    return common


def nsilr_mul (graph, all_graphs, layer_no, layers, nodes_all, psi_param = 0.5) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for CLP-MUl for link prediction in multiplex networks

    layers = int(layers)
    all_graphs[layer_no] = graph
    print("running nsilr mul sota for layer - "+str(layer_no)+
          " out of - "+str(layers))
    G = graph
    adj = nx.adjacency_matrix(graph).todense()
    r_layer = np.zeros(shape=(layers, layers))
    print("computing layer similarity")
    for i in range(layers):
        for j in range(layers):
            if i < j:
                num = 0
                denom1 = 0
                denom2 = 0
                for node1 in range(len(nodes_all)):
                    for node2 in range(len(nodes_all)):
                        if node1 < node2:
                            if all_graphs[i].has_edge(node1,node2) and all_graphs[j].has_edge(node1,node2):
                                num += 1
                            if all_graphs[i].has_edge(node1, node2): denom1 += 1
                            if all_graphs[j].has_edge(node1,node2): denom2 += 1
                r_layer[i][j] = 2 * num / (denom1+denom2)
            else:
                r_layer[i][j] = r_layer[j][i]
    print("relative layer relevance - "+str(r_layer))
    common = np.zeros(shape=(len(nodes_all), len(nodes_all)))
    print("before common matrix calculation")
    for node1 in range(len(nodes_all)):
        for node2 in range(len(nodes_all)):
            if node1 != node2 and not G.has_edge(node1,node2):
                try:
                    preds = nx.resource_allocation_index(all_graphs[layer_no],
                                                         [(node1, node2)])
                    for u, v, p in preds: common[node1][node2] = (1-psi_param)*p
                except:
                    common[node1][node2] = 0
                for i in range(layers):
                    if i != layer_no:
                        try:
                            preds = nx.resource_allocation_index(all_graphs[i],
                                                             [(node1, node2)])
                            for u, v, p in preds: common[node1][node2] += psi_param * p * \
                                                                          r_layer[layer_no][i]
                        except:
                            common[node1][node2] +=0
    return common



