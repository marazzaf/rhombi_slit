from firedrake import *
import sys
import numpy as np

mesh = Mesh('mesh_aux.msh')

V = FunctionSpace(mesh, "CG", 1)

sig_1 = Constant(1)
sig_2 = Constant(-1.5)
sigma = conditional(lt(x[0], Constant(0)), sig_1, sig_2)

#Complaince matrix
uu = Function(V, name='solution')
mu = 
Gamma = as_tensor(())

#Weak formulation
u = TrialFunction(V)
v = TestFunction(V)
a = sigma * inner(grad(u), grad(v)) * dx
L = Constant(0) * v * dx

#Dirichlet BC
x = SpatialCoordinate(mesh)
xi = 
bcs = [DirichletBC(V, xi, 1)]



# With the setup out of the way, we now demonstrate various ways of
# configuring the solver.  First, a direct solve with an assembled
# operator.::

solve(a == L, uu, bcs=bcs) #, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"})
out = File('out.pvd')
out.write(uu)

ref = File('ref.pvd')
vv = Function(V, name='ref')
vv.interpolate(xi)
ref.write(vv)

sys.exit()

#Newton solver
solve(a == 0, uu, bcs=bcs)
