from firedrake import *
from firedrake.output import VTKFile
import sys

#Mesh
N = 100 #320
L, H = 15, 15
mesh = RectangleMesh(N, N, L, H, diagonal='crossed', name='meshRef')

#Function Space
V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#Initial guess
v = TestFunction(V)
u = TrialFunction(V)

#Weak form
a = inner(grad(u), grad(v)) * dx
L = inner(Constant(0), v) * dx

#Dirichlet BC
val_max = 0.45
val_min = 0
x = SpatialCoordinate(mesh)
aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
xi_D = conditional(lt(x[1], H/2), aux2, aux1)
bcs = [DirichletBC(V, xi_D, 1), DirichletBC(V, xi_D, 2)]

#Solving initial guess
xi_init = Function(V)
solve(a == L, xi_init, bcs=bcs)

#init = VTKFile('IG.pvd')
#init.write(xi)

#material parameters
alpha = -.9
beta = 0.

#Compliance matrix
xi = Function(V) #just for derivations below
xi = variable(xi)
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = diff(mu1, xi)
mu2_p = diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))
epsilon = .5 #.5
Gamma += epsilon * Constant(((1j, 0), (0, 1j))) #Adding a complex dissipation
DGamma = diff(Gamma, xi)

#Homogeneous Dirichlet BC for increment
bcs = [DirichletBC(V, Constant(0), 1), DirichletBC(V, Constant(0), 2)]

# Newton iteration
tol = 1e-5
maxiter = 30
d_xi = Function(V, name='increment')
Xi = Function(V, name='real part xi')
xi_res= Function(V, name='xi')
xi_res.assign(xi_init)
for iter in range(maxiter):
    #Weak formulation
    Xi.interpolate(real(xi_res))
    Gamma_aux = replace(Gamma, {xi: Xi})
    DGamma_aux = replace(DGamma, {xi: Xi})
    a = inner(dot(Gamma_aux, grad(u)), grad(v)) * dx + inner(u * dot(DGamma_aux, grad(xi_res)), grad(v)) * dx

    #RHS
    L = -inner(dot(Gamma_aux, grad(xi_res)), grad(v)) * dx
    
    # compute the Newton increment by solving the linearized problem
    solve(a == L, d_xi, bcs=bcs)

    # update the solution
    xi_res.assign(xi_res + d_xi)

    #check increment size as convergence test
    eps = norm(d_xi, 'h1') 
    print('iteration{:3d}  H1-norm of delta: {:10.2e}'.format(iter+1, eps))
    if eps < tol:
      break

#Writing the result
final = VTKFile('test.pvd')
final.write(xi_res)


#Recovering global rotation
#Weak formulation
a = inner(grad(u), grad(v)) * dx
#rhs
Xi.interpolate(real(xi_res))
Gamma = as_tensor(((Constant(0), Gamma12,), (Gamma21, Constant(0))))
Gamma_aux = replace(Gamma, {xi: Xi})
l = inner(dot(Gamma_aux, grad(Xi)), grad(v)) * dx

gamma = Function(V, name='gamma')
nullspace = VectorSpaceBasis(constant=True)
solve(a == l, gamma, nullspace=nullspace)

rotation = VTKFile('non_pull_rot.pvd')
rotation.write(gamma)

#Recovering global disp
A = as_tensor(((mu1, Constant(0)), (Constant(0), mu2)))
A = replace(A, {xi:Xi})
R = as_tensor(((cos(gamma), -sin(gamma)), (sin(gamma), cos(gamma))))

W = VectorFunctionSpace(mesh, 'CG', 1) #2
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u), grad(v)) * dx
l = inner(dot(R, A), grad(v))  * dx

y = Function(W, name='yeff')
solve(a == l, y, nullspace=nullspace)

#disp = VTKFile('non_pull_disp.pvd')
##y.interpolate(as_vector((x[0], x[1])))
#disp.write(y)

## Saving the result
#with CheckpointFile("Ref.h5", 'w') as afile:
#    afile.save_mesh(mesh)
#    afile.save_function(y)
#sys.exit()

#Plotting results
import matplotlib.pyplot as plt
import numpy as np
original_coords = np.real(mesh.coordinates.dat.data)
deformed_coords = np.real(y.vector()[:]) #y_aux.dat.data
#x = original_coords[:,0]
#y = original_coords[:,1]
x = deformed_coords[:,0]
y = deformed_coords[:,1]
z = np.real(Xi.vector()[:])

# Create a triangulation
from matplotlib.tri import Triangulation
tri = Triangulation(x, y)

# Plot using tripcolor
#plt.figure(figsize=(8, 6))
plt.tripcolor(tri, z, cmap="jet", shading="flat") 
plt.colorbar(label="z = f(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = f(x, y) without Grid (Triangulation)")
plt.gca().set_aspect('equal')
plt.show()

#import scipy as sp
#X = np.linspace(min(x), max(x))
#Y = np.linspace(min(y), max(y))
#X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
#interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
#Z = interp(X, Y)
#plt.pcolormesh(X, Y, Z, shading='auto')
#plt.plot(x, y, "ok", label="input point")
#plt.legend()
#plt.colorbar()
#plt.axis("equal")
#plt.show()
