
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






