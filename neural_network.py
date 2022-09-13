from audioop import bias


import numpy as np

inputs =  [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 2

out = np.multiply(inputs, weights)
out = sum(out) + bias
print(out)