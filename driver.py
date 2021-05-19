import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_data():
    lines = []
    with open('input.txt') as f:
        lines = f.readlines()
    return lines


# define main method
if __name__ == '__main__':

    print_hi('K-means')
    the_lines = read_data()

    count = 0
    for line in the_lines:
        count += 1
        print(f'line {count}: {line}')

    x, y = make_blobs(centers=4, n_samples=500, n_features=2,shuffle=True,random_state=42)
    print(x.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(k=clusters, max_iterations=150, plot_steps=False)
    y_predict = k.compute_algorithm(x)
