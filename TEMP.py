# import tensorflow as tf
import numpy as np
import random
import munkres
# mat = np.random.randint(0,100,(5,5))
# print(mat)
#
# cols = np.argsort(mat, axis=1)[:,-3:]
# rows = np.array(range(5)).repeat(3)
# print(rows)
# print(cols.flatten())
# print()

# num_nodes = 5
# rows = np.array(range(num_nodes)).repeat(num_nodes)
# cols = np.array([list(range(num_nodes)) for _ in range(num_nodes)]).flatten()
#
# print(rows)

# def create_mask(shape, probability):
#     prob_value = probability * 1000
#     mask = np.random.randint(1, 1000, size= shape)
#     mask[mask < prob_value] = 0
#     mask[mask >= prob_value] = 1
#
#     return mask
#
# shape = (5, 5)  # 掩码的形状
# ones_ratio = 0.5  # 1的比例，例如0.5表示50%的1
# mask = create_mask(shape, ones_ratio)
#
# print(mask)

# def hungarian(X):
#     m = munkres.Munkres()
#     assignments = m.compute(-X)
#
#     P = np.zeros(shape = X.shape, dtype = X.dtype)
#     for row, col in assignments:
#         P[row][col] = 1.0
#     return P
#
# x = np.random.randint(0,100,(5,5))*1.0
# print(x)
# print(hungarian(x))
names = ["nug24", "nug30", "nug27", "nug18", "nug25", "nug22"]

names = sorted(names)

print(names)
