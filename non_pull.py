from firedrake import *
import sys

mesh = Mesh('mesh_test.msh')
L = 16
H = 16

V = FunctionSpace(mesh, "CG", 1)
print('Nb dof: %i' % V.dim())

#material parameters
alpha = -.9
beta = 0

#Complaince matrix
xi = Function(V, name='xi')
mu1 = cos(xi) - alpha*sin(xi)
mu1_p = -sin(xi) - alpha*cos(xi)
mu2 = cos(xi) + beta*sin(xi)
mu2_p = -sin(xi) + beta*cos(xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))
delta = 0.05 #how to get it automatically?

#Weak formulation
v = TestFunction(V)
a = inner(dot(Gamma, grad(xi)), dot(Gamma, grad(v))) * dx
#a = inner(dot(Gamma, grad(xi)), grad(v)) * dx
a += delta * inner(grad(xi), grad(v)) * dx

#Dirichlet BC
x = SpatialCoordinate(mesh)
val = 0.45
aux1 = val * (2 - x[1]/H*2)
aux2 = val * x[1]/H*2
#xi_D = conditional(lt(x[1], H/2), aux2, aux1)
#test
xi_D = -4*val/H**2 * x[1] * (x[1] - H)# + val
bcs = [DirichletBC(V, xi_D, 2)]

#xi.interpolate(Constant(0.1))

#Newton solver
solve(a == 0, xi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

final = File('non_pull_sol.pvd')
final.write(xi)

#poisson = File('non_pull_poisson.pvd')
#aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
#poisson.write(aux)

#test = File('test_pull.pvd')
#W = FunctionSpace(mesh, 'DG', 0)
#aux = interpolate(div(dot(Gamma, grad(xi))), W)
#test.write(aux)


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

disp = File('non_pull_disp.pvd')
disp.write(y)

