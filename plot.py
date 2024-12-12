import sys
import numpy as np
import matplotlib.pyplot as plt

#material parameters
alpha = -.9#-1.6
beta = .9 #0.4 #0 #.9

#Compliance matrix
xi_max = .8
xi_min = -.3
xi = np.arange(xi_min, xi_max, 1e-2)
mu1 = np.cos(xi) - alpha*np.sin(xi)
mu1_p = np.gradient(mu1, xi)
mu2 = np.cos(xi) + beta*np.sin(xi)
mu2_p = np.gradient(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
poisson = Gamma21*mu1**2/(Gamma12*mu2**2)

##plot
#plt.plot(xi, Gamma12, 'r-', label='$\Gamma_{12}$')
#plt.plot(xi, Gamma21, 'b-', label='$\Gamma_{21}$')
#plt.grid()
#plt.legend()
#plt.show()

#plot
plt.plot(xi, mu1**2, 'r-', label='$A_{11}$')
plt.plot(xi, mu2**2, 'b--', label='$A_{22}$')
plt.grid()
plt.legend()
plt.show()
