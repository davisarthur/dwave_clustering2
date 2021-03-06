import numpy as np
import random
import equalsize

'''
    distribution is an array of length k where each element is equal to the probability of a
    point being assigned to that many clusters or less
'''
def psuedosolution(N, k, target, distribution):
    psuedo = np.zeros(N * k).astype(int)
    for i in range(N):
        result = random.random()
        num_assigned = -1
        for j in range(len(distribution)):
            if result <= distribution[j]:
                num_assigned = j
                break
        possible_assignments = list(range(k))
        ideal_assignment = possible_assignments.index(target[i])
        possible_assignments.remove(ideal_assignment)
        
        if num_assigned == 0:
            continue
        if num_assigned >= 1:
            psuedo[i + N * ideal_assignment] = 1
            for _ in range(num_assigned - 1):
                new_assignment = random.choice(possible_assignments)
                possible_assignments.remove(new_assignment)
                psuedo[i + N * ideal_assignment] = 1
    return psuedo

N = 32
k = 2
X = np.array([[-0.82758983, 0.43967755],\
    [-1.68537408,  1.78473034],\
    [-2.05593747,  2.08133468],\
    [-0.47036195,  1.20097233],\
    [ 0.20189709, -1.45523064],\
    [ 0.90544759, -1.04361529],\
    [-0.82962297,  1.33268942],\
    [ 0.5147425 , -0.03127084],\
    [-1.01653031,  1.74477574],\
    [-0.1527354 ,  1.29022105],\
    [-1.0247734 ,  0.25337277],\
    [-0.97269   ,  0.17417726],\
    [-0.71861795,  1.09846808],\
    [-2.03028777,  2.91404947],\
    [-1.99034971,  2.12805128],\
    [ 0.21498273, -1.6738801 ],\
    [ 0.03193106, -0.89942596],\
    [ 1.77840496, -1.7249661 ],\
    [ 0.14668833, -0.05796377],\
    [ 0.59367115, -1.50148394],\
    [ 0.28042237, -0.5893913 ],\
    [-2.15931588,  2.15865488],\
    [ 0.91945456, -0.21094487],\
    [-0.92116131,  1.03614018],\
    [ 1.71836104, -3.41574584],\
    [ 2.35365471, -1.0390555 ],\
    [ 1.6111599 , -0.9143762 ],\
    [ 0.67663935, -1.57037733],\
    [ 0.60653457, -1.08551495],\
    [-0.37898001,  0.12791667],\
    [ 2.6220156 , -2.14575526],\
    [ 2.15174964, -3.24099123]])
target = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
ogdistribution = [0.4, 0.5, 0.1]
distribution = [0.0] * len(ogdistribution)
for i in range(len(ogdistribution)):
    distribution[i] = sum(ogdistribution[0:i + 1])
psuedo = psuedosolution(N, k, target, distribution)
postpsuedo = equalsize.postprocess_soph(X, psuedo)
print(psuedo)
print(postpsuedo[1])

                


    