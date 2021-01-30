import numpy as np
import time
import dimod
import equalsize
import balanced
import random
import scipy.spatial.distance
import reader
from sklearn.cluster import KMeans
from datetime import datetime
from dwave.system import DWaveSampler, EmbeddingComposite
from sklearn import datasets, metrics

##
# Davis Arthur
# ORNL
# A number of performance tests for the classical algorithms and quantum algorithm
# 6-19-2020
##

def gen_data(N, k, d, d_informative = None, sep = 1.0):
    ''' Generate synthetic classification dataset

    Args:
        N: Number of points
        k: Number of clusters
        d: dimension of each data point
        d_informative: number of features that are important to classification
        sep: factor used to increase or decrease seperation between clusters

    Returns:
        X: training data as numpy array
        labels: class assignment of each point
    '''
    if d_informative == None:
        d_informative = d
    return datasets.make_classification(n_samples=N, n_features=d, \
        n_informative=d_informative, n_redundant=0, n_classes=k, \
        n_clusters_per_class=1, flip_y=0.01, class_sep=sep)

def gen_iris(N, k):
    ''' Generate data set from Iris dataset

    Args:
        N: Number of points
        k: Number of clusters

    Returns:
        X: training data as numpy array
        target: class assignment of each point
    '''
    iris = datasets.load_iris()
    length = 150                    # total number of points in the dataset
    pp_cluster = 50                 # number of data points belonging to any given iris
    d = 4                           # iris dataset is of dimension 4
    data = iris["data"]             # all iris datapoints in a (150 x 4) numpy array
    full_target = iris["target"]    # all iris assignments in a list of length 150

    num_full = N % k        # number of clusters with maximum amount of entries
    available = [True] * length

    # build the data matrix and target list
    X = np.zeros((N, d))
    target = [-1] * N

    for i in range(k):
        for j in range(N // k):
            num = random.randint(0, pp_cluster - j - 1)
            count = 0
            for l in range(pp_cluster):
                if count == num:
                    X[i * (N // k) + j] = data[i * pp_cluster + l]
                    target[i * (N // k) + j] = full_target[i * pp_cluster + l]
                    break
                if available[l]:
                    count += 1

    for i in range(num_full):
        num = random.randint(0, pp_cluster - N // k - 1)
        count = 0
        for l in range(pp_cluster):
            if count == num:
                X[N // k * k + i] = data[i * pp_cluster + l]
                target[N // k * k + i] = full_target[i * pp_cluster + l]
                break
            if available[l]:
                count += 1
    return X, target

def test(X, target, N, k, filename, alpha = None, beta = None):
    # data file
    f = open(filename, "a")
    f.write(str(datetime.now()))    # denote date and time that test begins

    f.write("\nTarget: " + str(target))
    f.write("\n(N, k): " + "(" + str(N) + ", " + str(k) + ")")
    f.write("\nAlpha: " + str(alpha))
    f.write("\nBeta: " + str(beta))
    f.write("\nData: \n" + str(X))

    # solve classically
    balanced_solution = balanced.balanced_kmeans(X, k)[1]
    f.write("\nBalanced solution: " + str(balanced_solution))

    kmeans = KMeans(n_clusters=k).fit(X)
    sklearn_solution = kmeans.labels_
    f.write("\nSKlearn algorithm solution: " + str(sklearn_solution))

    # generate QUBO model
    model = equalsize.genModel(X, k, alpha = alpha, beta = beta)

    # find sampler
    sampler = equalsize.set_sampler()  # sets the D-Wave sampler

    # embed on the sampler
    start = time.time()
    embedding = equalsize.get_embedding(sampler, model)   # finds an embedding on the smapler
    end = time.time()
    f.write("\nTime to find embedding: " + str(end - start))

    start = time.time()
    embedded_model = equalsize.embed(sampler, model, embedding)   # embed on the D-Wave hardware
    end = time.time()
    f.write("\nTime to embed: " + str(end - start))
    f.write("\nNumber of qubits used: " + str(len(embedded_model.variables)))

    # get quantum solution
    start = time.time()
    embedded_solution_set = equalsize.run_quantum(sampler, embedded_model)    # run on the D-Wave hardware
    end = time.time()
    f.write("\nAnnealing time: " + str(end - start))
    start = time.time()
    solution = equalsize.dwave_postprocess(embedded_solution_set, embedding, model)
    end = time.time()
    f.write("\nD-Wave postprocessing time: " + str(end - start))
    f.write("\nSample set: " + str(solution))
    f.write("\n\n")
    f.close()

'''
    alphaupper4 = 1.0
    alphaupper3 = 0.5
    alphaupper2 = 0.25

    betaupper4 = 1.0
    betaupper3 = 0.5
    betaupper2 = 0.25

    problems4 = [(8, 4), (12, 4), (16, 4)]
    problems3 = [(15, 3), (18, 3), (21, 3)]
    problems2 = [(16, 2), (24, 2), (32, 2)]
'''
def alphabeta(N, k, X, target, max_alpha, max_beta, resolution = 8, fname = "data/revisions_ab.txt"):

    alpha_range = np.linspace(0.0, max_alpha, resolution)
    beta_range = np.linspace(0.0, max_beta, resolution)

    for alpha in alpha_range:
        for beta in beta_range:
            test(X, target, N, k, fname, alpha = alpha, beta = beta)

'''
    problem list
    synth: 
        (8, 2, 0.1, 1.4), (16, 2, 0.1, 1.4), (24, 2, 0.1, 1.4), (32, 2, 0.1, 1.2)
        (12, 3, 0.2, 1.2), (15, 3, 0.2, 1.2), (18, 3, 0.2, 0.6), (21, 3, 0.1, 1.0)
        (8, 4, 1.6, 0.2), (12, 4, 1.6, 0.1), (16, 4, 0.6, 0.1)
    iris:
        (8, 2, 0.1, 1.4), (16, 2, 0.1, 1.4), (24, 2, 0.1, 1.4), (32, 2, 0.1, 1.2)
        (9, 3, 0.2, 1.2), (12, 3, 0.2, 1.2), (15, 3, 0.2, 1.2), (18, 3, 0.2, 0.6), (21, 3, 0.1, 1.0)
'''
def final(problem, alpha, beta, gentype, num_trials = 80, d = 2, f = "data/final_revisions.txt"):
    for _ in range(num_trials):
        N = problem[0]
        k = problem[1]
        test(N, k, d = d, filename = f, alpha = alpha, beta = beta, data = gentype)  

if __name__ == "__main__":
    N, k, X, target = reader.readspecs()
    max_alpha = 1.0
    max_beta = 2.0
    alphabeta(N, k, X, target, max_alpha, max_beta)
