import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
import argparse
from sklearn.cluster import SpectralClustering


def EMD(dist1, dist2):
    norm_d1 = dist1/np.sum(dist1)
    norm_d2 = dist2/np.sum(dist2)

    dif = norm_d1 - norm_d2
    result = 0

    for i in range(len(dif)):
        result += abs(np.sum(dif[:i]))
    return result/(len(dist1)-1)


def show_graph_with_labels(adjacency_matrix):
    plt.figure()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    gr = nx.convert_node_labels_to_integers(gr, first_label=1)
    nx.draw(gr, node_size=500, with_labels=True)


def build_distance_matrix(grades):
    distance_matrix = np.zeros((len(grades), len(grades)))
    for i in range(len(grades)):
        for j in range(i+1, len(grades)):
            distance_matrix[i, j] = distance_matrix[j, i] = EMD(grades.iloc[i].to_numpy(
                dtype=np.float64), grades.iloc[j].to_numpy(dtype=np.float64))
    return distance_matrix


def merge_connected_components(connected_components):
    for i in range(len(connected_components)-1):
        for j in range(len(connected_components[i])):
            for k in range(len(connected_components)):
                if k == i:
                    k += 1
                    continue

                if connected_components[i][j] in connected_components[k]:
                    new_comp = [connected_components[a] for a in range(len(connected_components)) if a not in [
                        i, k]] + [list(set(connected_components[i]+connected_components[k]))]
                    return merge_connected_components(new_comp)
    return connected_components


def get_connected_components(distance_matrix, threshold):
    connected_components = [[j for j in range(i, len(
        distance_matrix)) if distance_matrix[i, j] < threshold] for i in range(len(distance_matrix))]
    return merge_connected_components(connected_components)


def create_adj_matrix(connected_components, shape):
    adj_matrix = np.zeros(shape)
    for component in connected_components:
        idx = component[0]
        adj_matrix[idx, component[:]] = 1
        adj_matrix[component[:], idx] = 1
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix


def extract_grades_from_frame(frame):
    # 8-20
    return frame[['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F, F+', 'W']]


def get_filter_for_gpa_range(gpa_range):
    return lambda d: d.loc[(d['gpa'] >= gpa_range[0]) & (d['gpa'] < gpa_range[1])]


def show_filtered_histogram(data, filter=lambda input_data: input_data.loc[(input_data['gpa'] >= 2.4) & (input_data['gpa'] < 2.6)]):
    if filter != None:
        data = filter(data)
    grades = extract_grades_from_frame(data)
    plt.figure()
    for i in range(len(grades)):
        plt.plot(grades.iloc[i].to_numpy())


if __name__ == '__main__':
    # Handle console arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='./data/grade_detail_g1000_sorted.csv',
                    required=False, help='path to input dataset')
    ap.add_argument('-t', '--threshold', default=.03, required=False,
                    help='distance threshold for pictured connected component graph')
    ap.add_argument('-e', '--enrollment', default=1000, required=False,
                    help='Minimum enrollment for classes to be examined, ignore classes with less students')
    ap.add_argument('-r', '--range', default='2.4,2.6', required=False,
                    help='range of gpas to consider')
    ap.add_argument('-p', '--plot', default='N',
                    required=False, help='Set to True if plots should be shown')
    ap.add_argument('-c', '--clusters', default=2, required=False,
                    help='Amount of clusters to be detected by the SpectralClustering Algorithm')
    args = vars(ap.parse_args())

    # Parse dataset
    data = pd.read_csv(args['dataset']).rename(columns=lambda x: x.strip())
    print(EMD([10, 0, 0], [0, 0, 10]))  # Max EMD

    # Ignore classes with too few students
    data = data.loc[pd.to_numeric(
        data['Official Enrollmt']) >= int(args['enrollment'])].fillna(0)

    # print(data)

    # Full dataset
    grades = extract_grades_from_frame(data)
    distance_matrix = build_distance_matrix(grades)

    # Dataset restricted by gpa parameter
    filter = get_filter_for_gpa_range(
        [float(i) for i in args['range'].split(',')])
    res_data = filter(data)
    # res_data.head()
    res_grades = extract_grades_from_frame(res_data)
    res_distance_matrix = build_distance_matrix(res_grades)

    print(res_distance_matrix)

    # Comparison restricted vs unrestricted EMD-Matrix mean value
    # For default gpa range: 0.0864 (unr.) vs  0.026 (res.)
    print(
        f'Mean of unrestricted distance matrix: {distance_matrix.mean()} vs restricted distance matrix: {res_distance_matrix.mean()}')

    max_distance_index = np.unravel_index(
        np.argmax(distance_matrix), distance_matrix.shape)
    print(f'Max Distance between (unrestricted): {max_distance_index}')

    a_distance_matrix = np.array(
        [np.mean(distance_matrix[i]) for i in range(len(grades))])
    print(np.sort(a_distance_matrix))

    # ListLinePlot-Part
    amount_connected_components = []
    for i in range(1, 479):
        amount_connected_components.append(
            len(get_connected_components(distance_matrix, threshold=i/10000)))

    plt.figure()
    plt.plot(amount_connected_components)

    # Graph Part
    connected_comp = get_connected_components(
        distance_matrix, threshold=float(args['threshold']))
    adj_matrix = create_adj_matrix(connected_comp, distance_matrix.shape)

    show_graph_with_labels(adj_matrix)

    show_filtered_histogram(res_data, None)

    if args['plot'].lower() == 'y':
        plt.show()  # Show all plots at the same time

    # SpectralClustering attempt
    clustering = SpectralClustering(n_clusters=int(args['clusters']),
                                    random_state=None, affinity='precomputed').fit(distance_matrix)
    print(
        f'Len Distance Matrix: {len(distance_matrix)}, len clustering labels: {len(clustering.labels_)}')

    print(clustering.labels_)

    sample_cluster = np.where(clustering.labels_ == 1)[0]
    print(sample_cluster)

    # Pay attention: If cluster contains only one element, this will fail!
    print(
        f'EMD between elements {sample_cluster[:1]}: {distance_matrix[sample_cluster[0],sample_cluster[1]]}')
    print(f'Minimum EMD: {np.min(distance_matrix[np.nonzero(distance_matrix)])}')