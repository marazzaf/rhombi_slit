from firedrake import *
import sys
import numpy as np

mesh = Mesh('mesh_aux_pull.msh')
L = 16
H = 16

V = FunctionSpace(mesh, "CG", 2)
print('Nb dof: %i' % V.dim())

#material parameters
ar = 1
lamda1 = .95
lamda2 = .95/ar
lamda3 = .05
lamda4 = .05/ar

alpha = ar*(lamda4 - lamda2)
beta = (lamda1 - lamda3)/ar

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
u = TrialFunction(V)
v = TestFunction(V)
a = inner(dot(Gamma, grad(u)), grad(v)) * dx
L = Constant(0) * v * dx

#Dirichlet BC
x = SpatialCoordinate(mesh)
aux1 = 0.74 * (2 - x[1]/H*2)
aux2 = 0.74 * x[1]/H*2
xi = conditional(lt(x[1], H/2), aux2, aux1)
bcs = [DirichletBC(V, xi, 1)]

#Newton solver
a = inner(dot(Gamma, grad(uu)), grad(v)) * dx
solve(a == 0, uu, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

final = File('sol_pull.pvd')
final.write(uu)

poisson = File('poisson_pull.pvd')
aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
poisson.write(aux)
