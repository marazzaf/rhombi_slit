from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#Mesh
Nx = 5
xcoords = np.linspace(-1,1,Nx)
Ny = Nx//2
ycoords = np.linspace(0,1,Ny)
mesh = RectangleMesh(Point(-1,0), Point(1,1), Nx, Ny, diagonal="crossed")

#Boundary
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)

V = FunctionSpace(mesh, "CG", 1)

x = SpatialCoordinate(mesh)
sig_1 = Constant(1)
sig_2 = Constant(-1.5)
sigma = conditional(lt(x[0], Constant(0)), sig_1, sig_2)

u = TrialFunction(V)
v = TestFunction(V)

a = sigma * inner(grad(u), grad(v)) * dx

xi_1 = ((x[0]+1)**2 - (2*sig_1+sig_2)*(x[0]+1) / (sig_1+sig_2)) * sin(pi*x[1])
xi_2 = sig_1 * (x[0] - 1) * sin(pi*x[1]) / (sig_1+sig_2)
xi = conditional(lt(x[0], Constant(0)), xi_1, xi_2)
#xi = Expression('x[0] < 0 ? xi_1 : xi_2', xi_1=xi_1, xi_2=xi_2, degree=3)

bcs = [DirichletBC(V, xi, boundaries, 0)]

L = -div(sigma*grad(xi)) * v * dx

uu = Function(V, name='solution')

# With the setup out of the way, we now demonstrate various ways of
# configuring the solver.  First, a direct solve with an assembled
# operator.::

solve(a == L, uu, bcs=bcs) #, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"})
out = File('out.pvd')
out.write(uu)

ref = File('ref.pvd')
vv = project(xi, V)
ref.write(vv)

err = errornorm(uu, vv, 'h1')
print('Error: %.2e' % err)
