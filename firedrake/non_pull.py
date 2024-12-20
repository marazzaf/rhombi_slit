from firedrake import *
import sys
from firedrake.petsc import PETSc
from firedrake.output import VTKFile
import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf

#Mesh
N = 100 #320
L, H = 15, 15
mesh = RectangleMesh(N, N, L, H, diagonal='crossed', name='meshRef')
h = CellDiameter(mesh)
h_max = assemble(h * dx) / L / H
print('h max: %.1e' % h_max)
#sys.exit()

#Function Space
V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#material parameters
alpha = -.9
beta = 0
epsilon = .5 #.3

#Compliance matrix
xi = Function(V, name='xi')
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = -sin(xi) - alpha*cos(xi)
mu2_p = -sin(xi) + beta*cos(xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))
Gamma += epsilon * Constant(((1j, 0), (0, 1j))) #Adding a complex dissipation

#Weak formulation
v = TestFunction(V)
a = inner(dot(Gamma, grad(xi)), grad(v)) * dx

#Dirichlet BC
val_max = 0.45
val_min = 0
x = SpatialCoordinate(mesh)
aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
xi_D = conditional(lt(x[1], H/2), aux2, aux1)
bcs = [DirichletBC(V, xi_D, 1), DirichletBC(V, xi_D, 2)]

#Newton solver
solve(a == 0, xi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

#Removing the complex part
xi.interpolate(real(xi))

#Writing the result
final = VTKFile('non_pull_xi.pvd')
final.write(xi)

## Saving the result
#with CheckpointFile("non_pull_xi.h5", 'w') as afile:
#    afile.save_mesh(mesh)
#    afile.save_function(xi)

##Plotting results
#fig, axes = plt.subplots()
#contours = tricontourf(xi, axes=axes, cmap="jet")
#axes.set_aspect("equal")
#fig.colorbar(contours)
#plt.show()

#Recovering global rotation
V = FunctionSpace(mesh, "CG", 1) #2
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
#rhs
Gamma = as_tensor(((Constant(0), Gamma12,), (Gamma21, Constant(0))))
l = inner(dot(Gamma, grad(xi)), grad(v)) * dx

gamma = Function(V, name='gamma')
nullspace = VectorSpaceBasis(constant=True)
solve(a == l, gamma, nullspace=nullspace)

rotation = VTKFile('non_pull_rot.pvd')
rotation.write(gamma)

#Recovering global disp
A = as_tensor(((mu1, Constant(0)), (Constant(0), mu2)))
R = as_tensor(((cos(gamma), -sin(gamma)), (sin(gamma), cos(gamma))))

W = VectorFunctionSpace(mesh, 'CG', 1) #2
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u), grad(v)) * dx
l = inner(dot(R, A), grad(v))  * dx

y = Function(W, name='yeff')
solve(a == l, y, nullspace=nullspace)

disp = VTKFile('non_pull_disp.pvd')
#y.interpolate(y - as_vector((x[0], x[1])))
disp.write(y)

#Plotting results
fig, axes = plt.subplots()
contours = tricontourf(y, axes=axes, cmap="jet")
axes.set_aspect("equal")
fig.colorbar(contours)
plt.show()

sys.exit()
