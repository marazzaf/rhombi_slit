import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('res.txt', delimiter=',', comments='#')

n = data[:,0]
err = data[:,1]
err2 = data[:,2]
err3 = data[:,3]

conv = 2 * np.log(err[:-1] / err[1:]) / np.log(n[1:] / n[:-1])
print(conv)

conv2 = 2 * np.log(err2[:-1] / err2[1:]) / np.log(n[1:] / n[:-1])
print(conv2)

conv3 = 2 * np.log(err3[:-1] / err3[1:]) / np.log(n[1:] / n[:-1])
print(conv3)
