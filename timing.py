import numpy as np
import random

def psuedosolution(N, k, target, std):
    psuedo = np.zeros(N * k).as_type(int)
    for i in range(N):
        rand = int(np.round(np.random.normal(scale = std)))
        if rand > k:
            rand = k
        if rand > 0:
            psuedo[i + N * target[i]] = 1
            for n in range(rand - 1):
                random.randint(1, k - 1)
                


    