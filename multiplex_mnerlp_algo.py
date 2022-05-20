import networkx as nx
import numpy as np
import math
import sys
import datetime


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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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


def mnerlp_mul (graph, type = 1, alpha = 0.2, beta = 1.0) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

    print("running mnerlp mul type - "+str(type))
    print("before actual mnerlp alpha - "+str(alpha)+ " beta - "+str(beta))
    G = graph
    adj = nx.adjacency_matrix(graph).todense()
    common = np.zeros(shape = (len(adj), len(adj)))
    ego_str = np.zeros(shape = (len(adj), len(adj)))
    for node1 in G:
        for node2 in G:
            if G.has_edge(node1,node2) == 0 :
                ego_str[node1][node2] = -1
    triangles = nx.triangles(G)
    for node in G:
        neighbors = G.neighbors(node)
        #first level for direct neighoburs, increase ego strength by 3
        for single in neighbors :
            ego_str[node][single] += 5 * G.edges[node,single]['weight']
            #ego_str[single][node] += 3
        #2nd level where edges between neighbours of node considered
        #print("triangles for node "+str(node)+" is = "+str(triangles[node]))
        neighbors1 = G.neighbors(node)
        for neighbor1 in neighbors1 :
            neighbors2 = G.neighbors(node)
            for neighbor2 in neighbors2 :
                if G.has_edge(neighbor1,neighbor2)  :
                    #print("change 2nd level")
                    ego_str[neighbor1][neighbor2] += 4 * G.edges[neighbor1,neighbor2]['weight']
                    #ego_str[neighbor2][neighbor1] += 2
        #3rd level where 2 hop edges wrt node are considered
        neighbors = G.neighbors(node)
        set_2hop = set()
        for single in neighbors :
            neighbors_2hop = G.neighbors(single)
            for neighbor_2hop in neighbors_2hop :
                if neighbor_2hop != node and not G.has_edge(neighbor_2hop,node):
                    set_2hop.add(neighbor_2hop)
                    #print("change 3rd level")
                    ego_str[single][neighbor_2hop] += 3 * G.edges[neighbor_2hop,single]['weight']
                    neighbors_3hop = G.neighbors(neighbor_2hop)
                    for far_single in neighbors_3hop :
                        if far_single != single and not G.has_edge(far_single,single):
                            ego_str[neighbor_2hop][far_single] += 1 * G.edges[neighbor_2hop,far_single]['weight']
                    #ego_str[neighbor_2hop][single] += 1
        for node1 in set_2hop :
            for node2 in set_2hop :
                if node1 != node2 and G.has_edge(node1,node2):
                    ego_str[node1][node2] += 2 * G.edges[node1,node2]['weight']

    #print("ego strength after counting")
    '''for node1 in G:
        for node2 in G:
            if ego_str[node1][node2] != -1 and ego_str[node1][node2] != 0:
                print(ego_str[node1][node2])'''
    no_nodes = len(adj)
    max = 0
    for node1 in G :
        for node2 in G :
            if max < ego_str[node1][node2]: max = ego_str[node1][node2]
            ego_str[node1][node2] = ego_str[node1][node2]/no_nodes
    print("max ego strength is "+str(max))
    '''for node1 in G:
        for node2 in G:
            ego_str[node1][node2] = ego_str[node1][node2] / max'''
    print("before centrality dict type - " + str(type) + " alpha - " + str(alpha) + " beta - " + str(beta))
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    if type == 1:
        centrality_dict = nx.closeness_centrality(G)
    elif type == 2:
        centrality_dict = nx.betweenness_centrality(G)
    elif type == 3:
        centrality_dict = dict()
        cc = np.zeros(len(adj))
        for node in graph:
            if graph.degree(node) > 1:
                cc[node] = triangles[node] / (graph.degree(node) * (graph.degree(node) - 1) / 2)
                neighbors1 = nx.neighbors(graph, node)
                neighbors2 = nx.neighbors(graph, node)
                element = 0
                for single1 in neighbors1:
                    for single2 in neighbors2:
                        if graph.has_edge(single1, single2):
                            element += graph.edges[node, single1]['weight'] + graph.edges[node, single2]['weight']
                neighbors = nx.neighbors(graph, node)
                denom = 0
                for single in neighbors:
                    denom += graph.edges[node, single]['weight']
                if denom > 0:
                    denom = denom * 2 / graph.degree(node)
                    element = element / denom
                    cc[node] = cc[node] * element
                else:
                    cc[node] = 0
            centrality_dict[node] = cc[node]
    elif type == 4:
        centrality_dict = nx.load_centrality(G)
    elif type == 5:
        centrality_dict = nx.katz_centrality_numpy(G)
    elif type == 6:
        centrality_dict = nx.harmonic_centrality(G)
    else:
        print("type of centrality not defined")
        sys.exit()
    print("after centrality dict type - " + str(type) + " alpha - " + str(alpha) + " beta - " + str(beta))
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    print("before non edge mnerlp type - " + str(type) + " alpha - " + str(alpha) + " beta - " + str(beta))
    for node1 in G :
        for node2 in G :
            if node1 <= node2 :
                common_neighbors = nx.common_neighbors(G,node1,node2)
                common[node1][node2] = 0
                for single in common_neighbors :
                    #numerator = ego_str[node1][single]*ego_str[single][node2]
                    numerator = ((ego_str[node1][single]+ego_str[single][node2])**alpha) * (centrality_dict[single]**beta)
                    denominator = 0
                    for neighbor in G.neighbors(single) :
                        denominator += (ego_str[single][neighbor]**alpha)*(centrality_dict[neighbor]**beta)
                    #denominator = denominator**2
                    denominator = denominator
                    if denominator != 0: common[node1][node2] += numerator/denominator
            else :
                common[node1][node2] = common[node2][node1]
    print("after all mnerlp type - "+str(type)+" alpha - " + str(alpha) + " beta - " + str(beta))
    return common


def madm_mul (graph, all_graphs, layer_no, layers, nodes_all) :
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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
    ##Author-Shivansh Mishra, IIT(BHU) Varanasi, code for MNERLP-MUl for link prediction in multiplex networks

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



