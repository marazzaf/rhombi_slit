import dolfinx
from mpi4py import MPI
import numpy as np
LL,H = 16,16
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [LL,H]], [10, 10])
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x:0.5 * (1 - x[1]/H) )
print(u_c.x.array.dtype)

from petsc4py import PETSc
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

import ufl
alpha = -.9
beta = 0.
uu = dolfinx.fem.Function(V, dtype=np.complex128)
uu.interpolate(lambda x:0*x[0] + 0.1+0j)
mu1 = ufl.cos(uu) - alpha*ufl.sin(uu)
mu1_p = -ufl.sin(uu) - alpha*ufl.cos(uu)
mu2 = ufl.cos(uu) + beta*ufl.sin(uu)
mu2_p = -ufl.sin(uu) + beta*ufl.cos(uu)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = ufl.as_tensor(((-Gamma21, 0.), (0., Gamma12)))
delta = 1e-2
Gamma += ufl.as_tensor(((delta*1j, 0), (0, delta*1j)))

#Bilinear form
v = ufl.TestFunction(V)
a = ufl.inner(ufl.dot(Gamma, ufl.grad(uu)), ufl.grad(v)) * ufl.dx

#Boundary conditions
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
#boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
bc1 = dolfinx.fem.dirichletbc(u_c, dofs_L) #boundary_dofs)
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], LL))
bc2 = dolfinx.fem.dirichletbc(u_c, dofs_R)

#Nonlinear problem
problem = dolfinx.fem.petsc.NonlinearProblem(a, uu, bcs=[bc1,bc2])
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(uu)
assert(converged)
print(f"Number of interations: {n:d}")

with dolfinx.io.XDMFFile(mesh.comm, "res.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    uu.name = "Rotation"
    xdmf.write_function(uu)
