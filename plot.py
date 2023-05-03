import sys
import numpy as np
import matplotlib.pyplot as plt

#material parameters
alpha = -.9
beta = 0.21 #0 #0.54 #0.21

#Compliance matrix
xi_max = np.pi/3
xi = np.arange(0, xi_max, 1e-2)
mu1 = np.cos(xi) - alpha*np.sin(xi)
mu1_p = np.gradient(mu1, xi)
mu2 = np.cos(xi) + beta*np.sin(xi)
mu2_p = np.gradient(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
poisson = Gamma21*mu1**2/(Gamma12*mu2**2)

#plot
plt.plot(xi, -Gamma12, 'r-', label='$\Gamma_{12}$')
plt.plot(xi, Gamma21, 'b-', label='$-\Gamma_{21}$')
plt.legend()
plt.show()
