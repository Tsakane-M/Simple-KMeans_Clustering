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
        self.centroids = [self.class_data[0], self.class_data[4], self.class_data[7]]

        # optimization
        # CREATE CLUSTERS

        # initialise empty list of list for clusters
        clusters = [[] for _ in range(self.k)]

        # 1. get the index of closest centroid
        # 1.1 iterate over the data
        for i in range(len(self.class_data)):
            distances = [3]*3
            for j in range(len(self.centroids)):
                # calculate euclidean distances for each sample in the data
                distances[j] = euclidean_distance(self.class_data, self.centroids, i, j)

                # find closest index
                closest_index = np.argmin(distances)

            # assign closest index to array that stores clusters
            clusters[closest_index].append(i)

            print(clusters)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)









            # put the current sample index in the closest cluster
            # clusters[centroid_index].append(i)
















































