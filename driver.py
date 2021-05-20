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
    # Closing file
    f.close()
    return lines


def process_input(my_lines):
    my_lines = my_lines[1:]
    count = 0

    # get sample size
    samples = len(my_lines)

    # create a 2D array to store our data
    rows, cols = (samples, 3)
    this_data = [[3] * cols] * rows
    int_data = [[3] * cols] * rows

    # print(this_data)

    # extract data values x,y from the lines
    for index in range(samples):
        this_data[index] = my_lines[index].split()

    print(f'length of data is {len(this_data)}')
    return this_data


# define main method
if __name__ == '__main__':

    print_hi('K-means')
    the_lines = read_data()
    data = process_input(the_lines)

    # print(len(data))
    k = KMeans(3, 150, len(data))

    k.compute_algorithm(data)
