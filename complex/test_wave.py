import dolfinx
from mpi4py import MPI
import numpy as np
N = 100
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [1,1]], [N, N], diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))

from petsc4py import PETSc
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

#Bilinear form
import ufl
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
Gamma = ufl.as_tensor(((1, 0), (0, -1)))
delta = 1e-5
Gamma = ufl.as_tensor(((1, 0), (0, -1 * (1-1j*delta))))
a = ufl.inner(ufl.dot(Gamma, ufl.grad(u)), ufl.grad(v)) * ufl.dx

#Linear form
truc = dolfinx.fem.Function(V, dtype=np.complex128)
truc.interpolate(lambda x: 0*x[0])
L = ufl.inner(truc, v) * ufl.dx

#Boundary conditions
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
u_1 = dolfinx.fem.Function(V, dtype=np.complex128)
u_1.interpolate(lambda x: 0*x[0])
bc1 = dolfinx.fem.dirichletbc(u_1, dofs_L)
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
u_2 = dolfinx.fem.Function(V, dtype=np.complex128)
u_2.interpolate(lambda x: x[1]*x[1])
bc2 = dolfinx.fem.dirichletbc(u_2, dofs_R)
dofs_B = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
u_3 = dolfinx.fem.Function(V, dtype=np.complex128)
u_3.interpolate(lambda x: 0*x[1])
bc3 = dolfinx.fem.dirichletbc(u_3, dofs_B)
dofs_T = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1))
u_4 = dolfinx.fem.Function(V, dtype=np.complex128)
u_4.interpolate(lambda x: x[0])
bc4 = dolfinx.fem.dirichletbc(u_4, dofs_T)
bcs = [bc1, bc2, bc3, bc4]
#bcs = [bc4]


#Linear problem
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
sol = problem.solve()

with dolfinx.io.XDMFFile(mesh.comm, "wave.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    sol.name = "Sol"
    xdmf.write_function(sol)
