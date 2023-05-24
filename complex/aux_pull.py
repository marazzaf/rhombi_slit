import dolfinx
from mpi4py import MPI
import numpy as np
LL,H = 16,16
N = 80 #160
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [LL,H]], [N, N])
num_cells = mesh.topology.index_map(2).size_local
h = dolfinx.cpp.mesh.h(mesh, 2, range(num_cells))
h = h.max()
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

from petsc4py import PETSc
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

import ufl
alpha = -.9
beta = 0.9
uu = dolfinx.fem.Function(V, dtype=np.complex128)
uu.interpolate(lambda x:0*x[0] + 0.1+0j)
mu1 = ufl.cos(uu) - alpha*ufl.sin(uu)
mu1_p = -ufl.sin(uu) - alpha*ufl.cos(uu)
mu2 = ufl.cos(uu) + beta*ufl.sin(uu)
mu2_p = -ufl.sin(uu) + beta*ufl.cos(uu)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = ufl.as_tensor(((-Gamma21, 0.), (0., Gamma12)))
delta = np.sqrt(h) #h #np.sqrt(h) #1e-2
Gamma += ufl.as_tensor(((delta*1j, 0), (0, delta*1j)))
#test
#Gamma = ufl.as_tensor(((Gamma21, 0.), (0., Gamma12)))

#Bilinear form
v = ufl.TestFunction(V)
a = ufl.inner(ufl.dot(Gamma, ufl.grad(uu)), ufl.grad(v)) * ufl.dx

#Boundary conditions
u_bc = dolfinx.fem.Function(V, dtype=np.complex128)
val_max = 0.8416
val_min = 0.0831
xi_u = lambda x: (val_max-val_min) * x[1]/H*2 + val_min #if np.where(x[1] < H/2)
xi_d =lambda x:  (val_max-val_min) * (2 - x[1]/H*2) + val_min
#xi = lambda x: -4*val/H**2 * x[1] * (x[1] - H)
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
tol = 1
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0) and np.any(x[1]  > H/2))
u_bc.interpolate(xi_u)
bc1 = dolfinx.fem.dirichletbc(u_bc, dofs_L)
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], LL))
u_bc.interpolate(xi)
bc2 = dolfinx.fem.dirichletbc(u_bc, dofs_R)

#Nonlinear problem
J = dolfinx.fem.form(ufl.derivative(a, uu))
problem = dolfinx.fem.petsc.NonlinearProblem(a, uu, bcs=[bc1,bc2], J=J)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(uu)
assert(converged)
print(f"Number of interations: {n:d}")

with dolfinx.io.XDMFFile(mesh.comm, "aux_%i_test.xdmf" % N, "w") as xdmf:
    xdmf.write_mesh(mesh)
    uu.name = "xi"
    xdmf.write_function(uu)
