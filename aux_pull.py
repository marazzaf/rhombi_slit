from firedrake import *
import sys
import numpy as np

mesh = Mesh('mesh.msh')
L = 16
H = 16

V = FunctionSpace(mesh, "CG", 2)
print('Nb dof: %i' % V.dim())

##material parameters
alpha = -.9
beta = 0

#Complaince matrix
uu = Function(V, name='solution')
mu1 = cos(uu) - alpha*sin(uu)
mu1_p = -sin(uu) - alpha*cos(uu)
mu2 = cos(uu) + beta*sin(uu)
mu2_p = -sin(uu) + beta*cos(uu)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))

#Weak formulation
v = TestFunction(V)
#a = inner(dot(Gamma, grad(uu)), grad(v)) * dx
a = inner(dot(Gamma, grad(uu)), dot(Gamma, grad(v))) * dx
a += sqrt(0.06) * inner(grad(uu), grad(v)) * dx


#Dirichlet BC
val = 0.45
x = SpatialCoordinate(mesh)
aux1 = val * (2 - x[1]/H*2)
aux2 = val * x[1]/H*2
#xi = conditional(lt(x[1], H/2), aux2, aux1)
xi = -4*val/H**2 * x[1] * (x[1] - H)
bcs = [DirichletBC(V, xi, 2)]

#Newton solver
solve(a == 0, uu, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

final = File('aux_pull_sol.pvd')
final.write(uu)

poisson = File('aux_pull_poisson.pvd')
aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
poisson.write(aux)
