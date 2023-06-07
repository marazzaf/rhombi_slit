from firedrake import *
import sys
sys.path.append('./experiments/')
from interpolate import BC
from firedrake.petsc import PETSc

mesh = Mesh('mesh.msh')
L, H = 15,15

V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

##material parameters
alpha = -.9
beta = 0.9

#Complaince matrix
xi = Function(V, name='xi')
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = diff(mu1, xi)
mu2_p = diff(mu2, xi)
#mu1_p = -sin(xi) - alpha*cos(xi)
#mu2_p = -sin(xi) + beta*cos(xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))

#Weak formulation
v = TestFunction(V)
a = inner(dot(Gamma, grad(xi)), grad(v)) * dx

#Dirichlet BC
val_max = 0.74 #0.8416
val_min = 0 #0.0831
x = SpatialCoordinate(mesh)
aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
xi_D = conditional(lt(x[1], H/2), aux2, aux1)
bcs = [DirichletBC(V, xi_D, 2)]

#test
W = VectorFunctionSpace(mesh, V.ufl_element())
X = interpolate(mesh.coordinates, W)
vec_coord = X.dat.data_ro
res_BC = Function(V)
res = BC(vec_coord[:,0],vec_coord[:,1])
res_BC.dat.data[:] = res
#bcs = [DirichletBC(V, res_BC, 2)]

#Exit BC
out_BC = File('bc.pvd')
out_BC.write(res_BC)

#Newton solver
solve(a == 0, xi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

final = File('aux_pull_xi.pvd')
final.write(xi)

#poisson = File('aux_pull_poisson.pvd')
#aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
#poisson.write(aux)

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

#rotation = File('rot.pvd')
#rotation.write(gamma)

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

disp = File('aux_pull_disp.pvd')
disp.write(y)
