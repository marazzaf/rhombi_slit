from firedrake import *
import sys
import numpy as np

N = 100
mesh = UnitSquareMesh(N, N, diagonal="crossed")

V = FunctionSpace(mesh, "CG", 3)
print('Nb dof: %i' % V.dim())

#Bilinear form
v = TestFunction(V)
u = TrialFunction(V)
Gamma = as_tensor(((1, 0), (0, -1)))
delta = 5e-1
Gamma += delta * as_tensor(((1, 0), (0, 1)))
a = inner(dot(Gamma, grad(u)), grad(v)) * dx

#Linear form
L = Constant(0) * v * dx

#Dirichlet BC
x = SpatialCoordinate(mesh)
bcs = [DirichletBC(V, Constant(0), 1), DirichletBC(V, Constant(0), 3), DirichletBC(V, x[1]*x[1], 2), DirichletBC(V, x[0], 4)]

#Newton solver
sol = Function(V, name='sol')
solve(a == L, sol, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

final = File('test_wave.pvd')
final.write(sol)
