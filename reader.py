import numpy as np

def readspecs(fname = "data/specs.txt"):
    f = open(fname, "r")
    N = int(f.readline().split(":")[-1])
    k = int(f.readline().split(":")[-1])
    f.readline()
    X = read_array(f)
    target = read_assignments(f)
    return N, k, X, target

def read_dict(f):
    dict_str = f.readline()
    colon_index = dict_str.find(":")
    return eval(dict_str[colon_index + 2:])

# Reads and returns a numpy array from a file.
# File must be on the first line of the array
def read_array(f):
    X_array = []
    while True:
        line = f.readline()
        last = False
        if "]]" in line:
            last = True
        point_array_str = line.split("[")[-1].split("]")[0].split()
        point_array = []
        for point in point_array_str:
            point_array.append(float(point))
        X_array.append(point_array)
        if last:
            break
    return np.array(X_array)

# Reads an assignment array from a file.
# File must be on the line with the assignment
def read_assignments(f):
    assignment_array = []
    for a in f.readline().split(":")[1].split("[")[-1].split("]")[0].split():
        assignment_array.append(int(a.split(",")[0]))
    return np.array(assignment_array)