import numpy as np
import json

file = '../../data/ppi_data.txt'
data = np.loadtxt(file, dtype=str, delimiter='\t')
np.random.shuffle(data)
np.random.shuffle(data)
print(len(data))
val = data[:10000, :]
train = data[10000:, :]
np.savetxt('../../data/train.txt', train, fmt='%s', delimiter='\t')
np.savetxt('../../data/val.txt', val, fmt='%s',  delimiter='\t')
