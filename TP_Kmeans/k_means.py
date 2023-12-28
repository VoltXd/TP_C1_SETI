import numpy as np
import matplotlib.pyplot as plt
import csv
import random

def euclidean_distance(u1, u2):
    if len(u1) != len(u2):
        raise Exception("euclidean_distance => Different dimensions : dim(u1) = {}, dim(u2) = {}".format(len(u1), len(u2)))
    distance = 0
    for i in range(len(u1)):
        distance += (u1[i] - u2[i])**2
    return np.sqrt(distance)

def find_closest_cluster(x, centroids):
    closest_cluster = 0
    closest_cluster_distance = euclidean_distance(x, centroids[0])

    for i in range(1, len(centroids)):
        cluster_distance = euclidean_distance(x, centroids[i])
        if cluster_distance < closest_cluster_distance:
            closest_cluster = i
            closest_cluster_distance = cluster_distance
    return closest_cluster

def plot_clusters(centroids, x_array, colors):
    for i, k in enumerate(centroids):
        plt.scatter(centroids[k][0], centroids[k][1], colors[i])
        plt.scatter(x_array[k, 0])

def plot_k_means(X, centroids, labels, COLORS_DICT):
    K = len(centroids)
    clustered_data = []
    for i in range(K):
        clustered_data.append([])

    for i in range(len(X)):
        clustered_data[labels[i]].append(X[i])

    for i in range(K):    
        plt.plot([x[0] for x in clustered_data[i]], [x[1] for x in clustered_data[i]], "+", label="Data Set", c=COLORS_DICT[i])
        plt.plot(centroids[i][0], centroids[i][1], "s", label="Centroids", c="black")
    plt.legend()
    plt.show()
    return

def k_means(X, K):
    # Get data borders
    DIMENSION = len(X[0])
    x_min = []
    x_max = []
    for i in range(DIMENSION):
        column = [x[i] for x in X]
        x_min.append(min(column))
        x_max.append(max(column))
    
    # CONSTANTS, Initial conditions, Stopping parameters
    MAX_ITERATIONS = 10000
    ENDING_VARIATION_DISTANCE = 0.0001

    iterations = 0
    maximum_variation_distance = np.inf

    centroids = []
    for i in range(K):
        centroids.append(np.random.uniform(x_min, x_max, None))
    labels = [None] * len(X)

    # Début itérations
    while ENDING_VARIATION_DISTANCE < maximum_variation_distance and iterations < MAX_ITERATIONS:
        # Update previous centroids
        previous_centroids = centroids.copy()

        # Classification ()
        for i in range(len(X)):
            labels[i] = find_closest_cluster(X[i], centroids)

        # Recentrage
        for i in range(K):
            centroids[i] = np.zeros(DIMENSION)
            number_of_elements = 0
            for j in range(len(X)):
                if i == labels[j]:
                    centroids[i] += X[j]
                    number_of_elements += 1
            centroids[i] /= number_of_elements

        # Update ending parameters
        iterations += 1
        maximum_variation_distance = max([euclidean_distance(previous_centroids[i], centroids[i]) for i in range(len(centroids))])
    
    print("Iterations : {}, Max variation : {}".format(iterations, maximum_variation_distance))

    # Classification FINALE
    for i in range(len(X)):
        labels[i] = find_closest_cluster(X[i], centroids)

    return centroids, labels

def main():
    K = 6
    COLORS_DICT = { 0 : "r", 1 : "b", 2 : "g", 3 : "c", 4 : "y", 5 : "m" }

    x_array = []
    
    # Retrieve the data
    with open('kmeans_6_classes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                x_array.append(np.array([float(element) for element in row]))
                line_count += 1

    # Compute K-Means
    centroids, labels = k_means(x_array, K)

    # Plot result
    plot_k_means(x_array, centroids, labels, COLORS_DICT)



if __name__ == "__main__":
    main()