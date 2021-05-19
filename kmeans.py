import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x2-x1)**2))
    return distance


class KMeans:
    def __init__(self, k=3, max_iterations=100, plot_steps=False):
        self.k = k
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps

        # list of lists(sample indices for each cluster)
        self.clusters = [[] for _ in range(self.k)]

        # mean feature vector for each cluster
        self.centroids = []

    def compute_algorithm(self, x):
        self.x= x
        self.n_samples, self.n_features =x.shape

        # initialise centroids
        random_sample_indexes=np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[index] for index in random_sample_indexes]

        # optimization
        for _ in range(self.max_iterations):

            # update clusters
            self.clusters = self.create_clusters(self.centroids)

            if self.plot_steps:
                self.plot

            # update centroids
            centroids_old = self.centroids()
            self.centroids = self.compute_centroids(self.clusters)

            if self.plot_steps:
                self.plot

            # check if converged
            if self.is_converged(centroids_old, self.centroids):
                break
        # return cluster labels
        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self, clusters):
        # stores the index of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index

            return labels

    def create_clusters(self, centroids):
        # initialise empty list of list for clusters
        clusters = [[] for _ in range(self.k)]

        # iterate over the data
        for i, sample in enumerate(self.x):
            # get the index of closest centroid
            centroid_index = self.closest_centroid(sample, centroids)

            # store old centroids to check for convergence later
            old_centroids = self.centroids

            # create new centroids with updated means
            self.centroids = self.compute_centroids(self.clusters)

            # put the current sample index in the closest cluster
            clusters[centroid_index].append(i)

        return clusters

    def closest_centroid(self, sample, centroids):
        # calculate distance from the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]

        # get min distance
        closest_index = np.argmin(distances)

        return closest_index

    def compute_centroids(self, clusters):
        # initialise the centroids with zeros
        centroids = np.zeros(self.k, self.n_features)

        # calculate the new mean for each cluster
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_index] = cluster_mean

        return centroids

    def is_converged(self, centroids_old, centroids):
        # calculate euclidean distances for each of the old vs new centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]

        if sum(distances == 0):
            converged = True
        else:
            converged = False

        return converged

    def plot(self,x):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()













































