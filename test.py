import numpy as np
import time
import dimod
import equalsize
import balanced
import random
import scipy.spatial.distance
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

def test(N, k, d = 2, filename = "revision_data.txt", alpha = None, beta = None, divisor = None, data = "synth"):
    # data file
    f = open(filename, "a")
    f.write(str(datetime.now()))    # denote date and time that test begins

    X = None
    target = None
    if data == "synth":
        X, target = gen_data(N, k, d)
        f.write("\nSynthetic data")
    if data == "iris":
        X, target = gen_iris(N, k)
        f.write("\nIris data")
    f.write("\nTarget: " + str(target))
    f.write("\n(N, k): " + "(" + str(N) + ", " + str(k) + ")")
    f.write("\nAlpha: " + str(alpha))
    f.write("\nBeta: " + str(beta))
    f.write("\nDivisor: " + str(divisor))
    f.write("\nData: \n" + str(X))

    # solve classically
    balanced_solution = balanced.balanced_kmeans(X, k)[1]
    f.write("\nBalanced solution: " + str(balanced_solution))
    f.write("\nBalanced objective value: " + str(equalsize.objective_value(X, balanced_solution, k)))

    kmeans = KMeans(n_clusters=k).fit(X)
    sklearn_solution = kmeans.labels_
    f.write("\nSKlearn algorithm solution: " + str(sklearn_solution))
    f.write("\nSKlearn algorithm objective value: " + str(equalsize.objective_value(X, sklearn_solution, k)))

    # generate QUBO model
    model = equalsize.genModel(X, k, alpha = alpha, beta = beta, divisor = divisor)

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

    # naive postprocessing
    centroids_naive, assignments_naive, num_viol_naive = \
        equalsize.postprocess_naive(X, solution)
    f.write("\nNaive assignments: " + str(assignments_naive))
    f.write("\nNaive objective value: " + str(equalsize.objective_value(X, assignments_naive, k)))

    # updated postprocessing
    centroids_soph, assignments_soph, num_viol_soph = \
        equalsize.postprocess_soph(X, solution)
    f.write("\nSophisticated assignments: " + str(assignments_soph))
    f.write("\nSophisticated objective value: " + str(equalsize.objective_value(X, assignments_soph, k)))
    f.write("\nNumber of violations: " + str(num_viol_soph))
    f.write("\n\n")
    f.close()

if __name__ == "__main__":
    divisor = 0.9
    d = 2
    left_alphas = [1.0 / 2**7, 1.0 / 2**6]
    bottom_betas = [1.0 / 2**7, 1.0 / 2**6]
    #old_alphas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    old_alphas = [4.0, 8.0]
    old_betas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

    num_trials = 3
    synth_configs = [(16, 2), (24, 2), (32, 2), (12, 3), (15, 3), (18, 3), (21, 3), (8, 4), (12, 4), (16, 4)]
    #for alpha in left_alphas:
    #    for beta in old_betas:
    #        for config in synth_configs:
    #            for _ in range(num_trials):
    #                N = config[0]
    #                k = config[1]
    #                test(N, k, d = d, filename = "revision_data_5.txt", alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")

    #for alpha in left_alphas:
    #    for beta in bottom_betas:
    #        for config in synth_configs:
    #            for _ in range(num_trials):
    #                N = config[0]
    #                k = config[1]
    #                test(N, k, d = d, filename = "revision_data_5.txt", alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")

    for alpha in old_alphas:
        for beta in bottom_betas:
            for config in synth_configs:
                for _ in range(num_trials):
                    N = config[0]
                    k = config[1]
                    test(N, k, d = d, filename = "revision_data_5.txt", alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")

    #betas = [4.0, 8.0]
    #synth_configs = [(8, 2), (16, 2), (24, 2), (32, 2), (12, 3), (15, 3), (18, 3), (21, 3), (8, 4), (12, 4), (16, 4)]
    #iris_configs = [(8, 2), (16, 2), (24, 2), (32, 2), (9, 3), (12, 3), (15, 3), (18, 3), (21, 3)]
    
    #spare_synth_configs = [(32, 2), (12, 3), (15, 3), (18, 3), (21, 3), (8, 4), (12, 4), (16, 4)]
    #for config in spare_synth_configs:
    #    alpha = 1.0
    #    beta = 2.0
    #    N = config[0]
    #    k = config[1]
    #    test(N, k, d = d, filename = "revision_data_5.txt", \
    #        alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")

    #for config in iris_configs:
    #    alpha = 1.0
    #    beta = 2.0
    #    N = config[0]
    #    k = config[1]
    #    test(N, k, d = d, filename = "revision_data_5.txt", \
    #        alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "iris")

    #spare_alphas = [2.0, 4.0, 8.0]
    #for alpha in spare_alphas:
    #    beta = 2.0
    #    for config in synth_configs:
    #        N = config[0]
    #        k = config[1]
    #        test(N, k, d = d, filename = "revision_data_5.txt", \
    #            alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")
    #    for config in iris_configs:
    #        N = config[0]
    #        k = config[1]
    #        test(N, k, d = d, filename = "revision_data_5.txt", \
    #            alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "iris")

    #for beta in betas:
    #    for alpha in alphas:
    #        for config in synth_configs:
    #            N = config[0]
    #            k = config[1]
    #            test(N, k, d = d, filename = "revision_data_5.txt", \
    #                alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "synth")
    #        for config in iris_configs:
    #            N = config[0]
    #            k = config[1]
    #            test(N, k, d = d, filename = "revision_data_5.txt", \
    #                alpha = alpha * N / k, beta = beta * N / k, divisor = divisor, data = "iris")

