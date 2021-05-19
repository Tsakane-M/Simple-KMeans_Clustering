import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_data():
    lines = []
    with open('input.txt') as f:
        lines = f.readlines()
    return lines


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x2-x1)**2))
    return distance


class KMeans:
    def __init__(self, k=3, max_iterations=100, plot_steps=false):
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
        self.centroids = [1, 4, 7]

        # optimization
        for _ in range(self.max_iterations):

            # update clusters
            self.clusters = self.create_clusters(self.centroids)

            # update centroids
            centroids_old = self.centroids()
            self.centroids = self.compute_centroids(self.clusters)

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
                labels




    def create_clusters(self, centroids):
        # initialise empty list of list for clusters
        clusters = [[] for _ in range(self, k)]

        # iterate over the data
        for index, sample in enumerate(self, x):
            # get the index of closest centroid
            centroid_index = self.closest_centroid(sample, centroids)

            # store old centroids to check for convergence later
            old_centroids = self.centroids

            # create new centroids with updated means
            self.centroids = self.compute_centroids(self.clusters)

            # put the current sample index in the closest cluster
            clusters[centroid_index].append(index)

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
            converged = true
        else:
            converged = false

        return converged





































    # define main method
if __name__ == '__main__':

    print_hi('K-means')
    the_lines = read_data()

    count = 0
    for line in the_lines:
        count += 1
        print(f'line {count}: {line}')






