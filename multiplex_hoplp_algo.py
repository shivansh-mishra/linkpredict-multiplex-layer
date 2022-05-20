import networkx as nx
import numpy as np
import math


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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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


def hoplp_mul (graph, param=0.01, reg_inf_arr=[3,4]):
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

    print("hoplp mul")
    adj = nx.adjacency_matrix(graph).todense()
    score = np.zeros((len(adj), len(adj)))
    prev = np.zeros((len(adj), len(adj)))
    node_wt = np.zeros(len(adj))
    for node in graph:
        for single in graph.neighbors(node):
            node_wt[node] += graph.edges[node,single]['weight']
    max = -99999999
    for node in graph:
        if node_wt[node] > max:
            max = node_wt[node]
    node_wt = node_wt / max
    #for path length 2, 1 neighbour betwwen node 1 and node 2
    for node1 in graph :
        for node2 in graph :
            common_neighbours_all = nx.common_neighbors(graph,node1,node2)
            for common_neighbour in common_neighbours_all :
                if node_wt[common_neighbour] != 0 :
                    score[node1][node2] = score[node1][node2] + math.log(1 / node_wt[common_neighbour])
            prev[node1][node2] = score[node1][node2]
    for p in reg_inf_arr:
        temp = np.zeros((len(adj), len(adj)))
        for node1 in graph :
            for node2 in graph :
                temp1 = 0
                for node_intermediate in graph.neighbors(node1):
                    if prev[node_intermediate][node2] != 0 and graph.has_edge(node_intermediate,node2):
                        curr = score[node1][node_intermediate] * prev[node_intermediate][node2] * ((param) ** (p-2))
                        temp1 = temp1 + curr
                temp[node1][node2] = temp1
        for node1 in graph:
            for node2 in graph:
                score[node1][node2] = score[node1][node2] + temp[node1][node2]
                prev[node1][node2] = temp[node1][node2]
    return score


def madm_mul (graph, all_graphs, layer_no, layers, nodes_all) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for HOPLP-MUl for link prediction in multiplex networks

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



