import numpy as np


class KMeans:
    def __init__(self, k=5, max_iterations=100, plot_steps=False, sample_size=10, class_data=None):
        if class_data is None:
            class_data = [[3] * 3] * 2
        self.k = k
        self.max_iterations = max_iterations
        self.plot_steps = plot_steps
        self.sample_size = sample_size
        self.class_data = class_data

        # list of lists(sample indices for each cluster)
        self.clusters = [[] for _ in range(self.k)]

        # mean feature vector for each cluster (x1,y1) , (x2,y2), (x3,y3)
        self.centroids = [[0, 0], [0, 0], [0, 0]]

    def compute_algorithm(self, class_dat):
        # initialise centroids indices as 1 ,4, 7
        self.class_data = class_dat
        self.centroids = [[int(self.class_data[0][1]), int(self.class_data[0][2])], [int(self.class_data[3][1]), int( self.class_data[3][2])], [int(self.class_data[6][1]), int(self.class_data[6][2])]]

        # optimization
        # CREATE CLUSTERS

        # open file for output
        f = open("output.txt", "a")

        # clear file
        f.truncate(0)

        # Optimize clusters
        for it in range(self.max_iterations):
            print(f'Iteration {it+1}', file=f)
            print(f'...........', file=f)
            # initialise empty list of list for clusters
            clusters = [[] for _ in range(self.k)]

            # 1. get the index of closest centroid
            # 1.1 iterate over the data
            for i in range(8):
                distances = [3]*3
                # calculate euclidean distances for each sample in the data
                # print(self.class_data[i][1])
                # get x and y value from input
                x_data = self.class_data[i][1]
                y_data = self.class_data[i][2]

                for j in range(3):
                    # get x and y from centroids
                    x_cent = self.centroids[j][0]
                    y_cent = self.centroids[j][1]

                    # calculate euclidean distance
                    xSquare = (int(x_cent) - int(x_data)) ** 2
                    ySquare = (int(y_cent) - int(y_data)) ** 2
                    distance = np.sqrt(xSquare + ySquare)
                    distances[j] = distance
                # print(distances)
                # find closest index
                closest_index = np.argmin(distances)

                # assign closest index to array that stores clusters
                clusters[closest_index].append(i)
            self.clusters = clusters

            # Calculate new centroids from the clusters

            # save old centroids
            centroids_old = self.centroids

            # compute new centroids
            self.centroids = self.compute_centroids(self.clusters)

            # compute labels for the cluster contents
            the_labels = self.get_cluster_labels(self.clusters)

            # print centroids
            for c in range(3):
                print(f'Cluster  {c + 1}: {the_labels[c]}', file=f)
                print(f'Centroid {c+1}: {centroids_old[c]}\n', file=f)
            print(f'\n', file=f)

            # check if clusters have changed
            if self.is_converged(centroids_old, self.centroids):
                print(f'Converged!', file=f)
                break
                # close file

        # Classify samples as the index of their clusters
        return the_labels

    def compute_centroids(self, clusters):
        # initialise the centroids with zeros
        centroids = [[0, 0], [0, 0], [0, 0]]

        for i in range(3):
            x_cluster_sum = 0
            y_cluster_sum = 0
            # print(f'Cluster {i}')
            for j in range(len(clusters[i])):
                xvalue = int((self.class_data[clusters[i][j]])[1])
                yvalue = int((self.class_data[clusters[i][j]])[2])
                # print(f'x is {xvalue} : y is {yvalue}')
                x_cluster_sum = x_cluster_sum + xvalue
                y_cluster_sum = y_cluster_sum + yvalue
            # calculate means
            x_cluster_mean = x_cluster_sum/len(clusters[i])
            y_cluster_mean = y_cluster_sum / len(clusters[i])

            # print(f'Cluster {i} xmean= {x_cluster_mean}')
            # print(f'Cluster {i} ymean= {y_cluster_mean}')
            # print(f'\n')

            # assign new means to centroid
            centroids[i][0] = x_cluster_mean
            centroids[i][1] = y_cluster_mean

        return centroids

    def is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, for all centroids
        distances = [0]*3
        for i in range(3):
            x_cent_old = int(centroids_old[i][0])
            y_cent_old = int(centroids_old[i][1])

            x_cent = int(centroids[i][0])
            y_cent = int(centroids[i][1])

            distance_x = x_cent_old - x_cent
            distance_y = y_cent_old - y_cent

            distance = np.sqrt(distance_x**2 + distance_y**2)
            distances[i] = distance
            # print(distances)
        summation = 0

        for i in range(3):
            summation = summation+distances[i]
            # print(summation)
        return summation == 0

    def get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = clusters
        # print(labels)

        for i in range(3):
        
            for j in range(len(clusters[i])):
                labels[i][j] = int((self.class_data[clusters[i][j]])[0])

        # print(labels)

        return labels
