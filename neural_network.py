from audioop import bias


import numpy as np

inputs =  [1.0, 2.0, 3.0, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87] 
bias1 = 2
bias2 = 3
bias3 = 0.5


out = [sum(np.multiply(inputs, weights1)) + bias1, 
    sum(np.multiply(inputs, weights2)) + bias2,
    sum(np.multiply(inputs, weights3)) + bias3]
print(out)