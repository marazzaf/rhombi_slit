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
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0+0j))
a = ufl.inner(ufl.dot(Gamma, ufl.grad(u)), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
#boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
bc1 = dolfinx.fem.dirichletbc(u_c, dofs_L) #boundary_dofs)
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], LL))
bc2 = dolfinx.fem.dirichletbc(u_c, dofs_R)
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc1,bc2])
uh = problem.solve()

with dolfinx.io.XDMFFile(mesh.comm, "res.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    uh.name = "Rotation"
    xdmf.write_function(uh)
