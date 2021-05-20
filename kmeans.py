import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(points1, points2, a, b):

    # calculate distance from the current sample to each centroid
    xSquare = (int(points1[a][1]) - int(points2[b][1])) ** 2
    ySquare = (int(points1[a][2]) - int(points2[b][2])) ** 2
    distance = np.sqrt(xSquare + ySquare)
    #print(distance)
    return distance


class KMeans:
    def __init__(self, k=5, max_iterations=100, plot_steps=False, sample_size=10, class_data=[[3] * 3] * 2):
        self.k = k
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps
        self.sample_size = sample_size
        self.class_data = class_data

        # list of lists(sample indices for each cluster)
        self.clusters = [[] for _ in range(self.k)]

        # mean feature vector for each cluster
        self.centroids = []

    def compute_algorithm(self, class_dat):
        # initialise centroids indices as 1 ,4, 7
        self.class_data = class_dat
        self.centroids = [self.class_data[0], self.class_data[3], self.class_data[6]]

        # optimization
        # CREATE CLUSTERS

        # initialise empty list of list for clusters
        clusters = [[] for _ in range(self.k)]

        # 1. get the index of closest centroid
        # 1.1 iterate over the data
        for i in range(8):
            distances = [3]*3
            for j in range(3):
                # calculate euclidean distances for each sample in the data
                distances[j] = euclidean_distance(self.class_data, self.centroids, i, j)

                # find closest index
                closest_index = np.argmin(distances)

            # assign closest index to array that stores clusters
            clusters[closest_index].append(i)
        self.clusters = clusters


        # Calculate new centroids from the clusters

        # save old centroids
        centroids_old = self.centroids

        self.centroids = self.compute_centroids(self.clusters)

    def compute_centroids(self, clusters):
        # initialise the centroids with zeros
        centroids = [0, 0, 0]
        print(clusters)

        for i in range(3):
            x_cluster_sum = 0
            y_cluster_sum = 0
            print(f'Cluster {i}')
            for j in range(len(clusters[i])):
                xvalue = int((self.class_data[clusters[i][j]])[1])
                yvalue = int((self.class_data[clusters[i][j]])[2])
                print(f'x is {xvalue} : y is {yvalue}')
                x_cluster_sum = x_cluster_sum + xvalue
                y_cluster_sum = y_cluster_sum + yvalue
            x_cluster_mean = x_cluster_sum/len(clusters[i])
            y_cluster_mean = y_cluster_sum / len(clusters[i])

            print(f'Cluster {i} xmean= {x_cluster_mean}')
            print(f'Cluster {i} ymean= {y_cluster_mean}')
            print(f'\n')
        return [0, 3, 6]
        # calculate the new mean for each cluster
        # for cluster_index, cluster in enumerate(clusters):
            # cluster_mean = np.mean(self.x[cluster], axis=0)
            # centroids[cluster_index] = cluster_mean\

















































