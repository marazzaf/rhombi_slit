import numpy as np
import matplotlib.pyplot as plt
import sys

#Load data
data = np.loadtxt('aux_pull.txt', comments='#')

#xi = data[:,2]
#print(data[data[:,0] > 2.25])
#sys.exit()

#Plot data
plt.scatter(data[:,0], data[:,1], c=data[:,2])
plt.colorbar()
#plt.xlim(0.5,2.4)
#plt.ylim(0.3,1.9)
plt.show()
#sys.exit()

#Computing the convex Hull
from scipy.spatial import ConvexHull
points = data[:,:2]
hull = ConvexHull(points)

#Plotting the Hull
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()
#sys.exit()

#Creating the Linear interpolation
from scipy.interpolate import LinearNDInterpolator
interp = LinearNDInterpolator(data[:,:2], data[:,2])
#modify that with position in ref config?
#thus change #data[:,:2]...

#Getting the BC
x = points[hull.vertices,0]
y = points[hull.vertices,1]
res = np.array([x,y, interp(x,y)]).T
print(res)
sys.exit()

#Plotting the linear interpolation
x = data[:,3]
X = np.linspace(min(x), max(x))
y = data[:,4]
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)
Z = interp(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto', cmap='jet')
#plt.plot(x, y, "ok", label="input point")
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.show()




















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































