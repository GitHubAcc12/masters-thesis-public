import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import argparse


def EMD(dist1, dist2):
    norm_d1 = dist1/np.sum(dist1)
    norm_d2 = dist2/np.sum(dist2)

    dif = norm_d1 - norm_d2

    inner_sum = lambda arr, end: np.sum(arr[:end])
    result = 0

    for i in range(len(dif)):
        result += abs(inner_sum(dif, i))
    return result/(len(dist1)-1)

def show_graph_with_labels(adjacency_matrix, mylabels):
    plt.figure()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    

def build_distance_matrix(grades):
    distance_matrix = np.zeros((len(grades), len(grades)))
    for i in range(len(grades)):
        for j in range(i+1, len(grades)):
            distance_matrix[i, j] = distance_matrix[j, i] = EMD(grades.iloc[i].to_numpy(dtype=np.float64), grades.iloc[j].to_numpy(dtype=np.float64))
    return distance_matrix


def merge_connected_components(connected_components):
    for i in range(len(connected_components)-1):
        for j in range(len(connected_components[i])):
            for k in range(len(connected_components)):
                if k == i:
                    k += 1
                    continue
            
                if connected_components[i][j] in connected_components[k]:
                    new_comp = [connected_components[a] for a in range(len(connected_components)) if a not in [i,k]] + [list(set(connected_components[i]+connected_components[k]))]
                    return merge_connected_components(new_comp)
    return connected_components


def get_connected_components(distance_matrix, threshold):
    connected_components = [[j for j in range(i, len(distance_matrix)) if distance_matrix[i, j] < threshold] for i in range(len(distance_matrix))]
    return merge_connected_components(connected_components)
    

def create_adj_matrix(connected_components, shape):
    adj_matrix = np.zeros(shape)
    for component in connected_components:
        idx = component[0]
        adj_matrix[idx, component[:]] = 1
        adj_matrix[component[:], idx] = 1
    return adj_matrix

    
    
if __name__=='__main__':
    # Handle console arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='./data/grade_detail_g1000_sorted.csv', required=False, help='path to input dataset')
    ap.add_argument('-t', '--threshold', default=.02, required=False, help='distance threshold for pictured connected component graph')
    args = vars(ap.parse_args())

    # Parse dataset
    data = pd.read_csv(args['dataset'])
    grades = data.iloc[:,8:20]
    print(EMD([10,0,0], [0,0,10])) # Max EMD
    distance_matrix = build_distance_matrix(grades)
    max_distance_index = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    print(f'Max Distance between: {max_distance_index}')

    a_distance_matrix = np.array([np.mean(distance_matrix[i]) for i in range(len(grades))])
    print(np.sort(a_distance_matrix))



    # ListLinePlot-Part
    amount_connected_components = []
    for i in range(1, 479):
        amount_connected_components.append(len(get_connected_components(distance_matrix, threshold=i/10000)))

    plt.figure()
    plt.plot(amount_connected_components)



    # Graph Part    
    connected_comp = get_connected_components(distance_matrix, threshold=float(args['threshold']))
    adj_matrix = create_adj_matrix(connected_comp, distance_matrix.shape)

    with open('label_dict.json', 'r') as jsonfile:
        labels = json.load(jsonfile)
    
    labels_cp = {} # Create a copy, because the keys need to be integers
    for key, value in labels.items():
        labels_cp.update({int(key)-1:value})

    show_graph_with_labels(adj_matrix, labels_cp)


    plt.show() # Show all plots at the same time
