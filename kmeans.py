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

    def cluster_samples(self, X):
        self.X = X
        self.n_samples, self.n_features =X.shape

        # initialise centroids
        selected_samples=[1,4,7]

            #optimization
            for _ in range(self.max_iterations):

                # update clusters

                # update centroids

                # check if converged
        #return cluster labels

















# define main method
if __name__ == '__main__':

    print_hi('K-means')
    the_lines = read_data()

    count = 0
    for line in the_lines:
        count += 1
        print(f'line {count}: {line}')






