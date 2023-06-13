import dolfinx
from mpi4py import MPI
import numpy as np
import sys
LL,H = 15,15
N = 320 #320 #160 #80
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [LL,H]], [N, N], diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
num_cells = mesh.topology.index_map(2).size_local
h = dolfinx.cpp.mesh.h(mesh, 2, range(num_cells))
h = h.max()
print('Size mesh: %.3e' % h)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
print('Nb dof: %i' % (V.dofmap.index_map.size_global * V.dofmap.index_map_bs))

from petsc4py import PETSc
assert np.dtype(PETSc.ScalarType).kind == 'c'

import ufl
alpha = -.9
beta = 0.
xi = dolfinx.fem.Function(V, dtype=np.complex128)
#xi.interpolate(lambda x:0*x[0] + 0.1+0j)
mu1 = ufl.cos(xi) - alpha*ufl.sin(xi)
mu2 = ufl.cos(xi) + beta*ufl.sin(xi)
mu1_p = ufl.diff(mu1, xi)
mu2_p = ufl.diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = ufl.as_tensor(((-Gamma21, 0.), (0., Gamma12)))
delta = np.sqrt(h) #h #np.sqrt(h) #1e-2
Gamma += ufl.as_tensor(((delta*1j, 0), (0, delta*1j)))

#Bilinear form
v = ufl.TestFunction(V)
a = ufl.inner(ufl.dot(Gamma, ufl.grad(xi)), ufl.grad(v)) * ufl.dx

#Boundary conditions
val_max = 0.45 #0.8416
val_min = 0 #0.0831
x = ufl.SpatialCoordinate(mesh)
aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
xi_D = ufl.conditional(ufl.lt(x[1], H/2), aux2, aux1)
xi_D = dolfinx.fem.Expression(xi_D, V.element.interpolation_points())
u_bc = dolfinx.fem.Function(V, dtype=np.complex128)
u_bc.interpolate(xi_D)

#Applying BC
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
u_bc.interpolate(xi_D)
bc1 = dolfinx.fem.dirichletbc(u_bc, dofs_L)
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], LL))
bc2 = dolfinx.fem.dirichletbc(u_bc, dofs_R)

#Nonlinear problem
#J = dolfinx.fem.form(ufl.derivative(a, uu))
#u = ufl.TrialFunction(V)
#J = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
problem = dolfinx.fem.petsc.NonlinearProblem(a, xi, bcs=[bc1,bc2])#, J=J)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(xi)
assert(converged)
print(f"Number of interations: {n:d}")

with dolfinx.io.XDMFFile(mesh.comm, "non_%i_xi.xdmf" % N, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xi.name = "xi"
    xdmf.write_function(xi)


#Recovering global rotation
u = ufl.TrialFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
#rhs
Gamma = ufl.as_tensor(((0, Gamma12,), (Gamma21, 0)))
L = ufl.inner(ufl.dot(Gamma, ufl.grad(xi)), ufl.grad(v)) * ufl.dx

# Assemble system
A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
A.assemble()
b = dolfinx.fem.petsc.create_vector(dolfinx.fem.form(L))
with b.localForm() as b_loc:
            b_loc.set(0)
dolfinx.fem.petsc.assemble_vector(b,dolfinx.fem.form(L))

# Create Krylov solver
solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)

# Create vector that spans the null space
nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
A.setNullSpace(nullspace)

# orthogonalize b with respect to the nullspace ensures that 
# b does not contain any component in the nullspace
nullspace.remove(b)

# Finally we are able to solve our linear system ::
gamma = dolfinx.fem.Function(V)
solver.solve(b,gamma.vector)

#Recovering global disp
A = ufl.as_tensor(((mu1, 0), (0, mu2)))
R = ufl.as_tensor(((ufl.cos(gamma), -ufl.sin(gamma)), (ufl.sin(gamma), ufl.cos(gamma))))

W = dolfinx.fem.VectorFunctionSpace(mesh, ('CG', 1)) #2
u = ufl.TrialFunction(W)
v = ufl.TestFunction(W)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(ufl.dot(R, A), ufl.grad(v)) * ufl.dx

# Assemble system
A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
A.assemble()
b = dolfinx.fem.petsc.create_vector(dolfinx.fem.form(L))
with b.localForm() as b_loc:
            b_loc.set(0)
dolfinx.fem.petsc.assemble_vector(b,dolfinx.fem.form(L))

# Create Krylov solver
solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)

# Create vector that spans the null space
nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
A.setNullSpace(nullspace)

# orthogonalize b with respect to the nullspace ensures that 
# b does not contain any component in the nullspace
nullspace.remove(b)

# Finally we are able to solve our linear system ::
y = dolfinx.fem.Function(W)
solver.solve(b,y.vector)

#Writing output
with dolfinx.io.XDMFFile(mesh.comm, "non_%i_disp.xdmf" % N, "w") as xdmf:
    xdmf.write_mesh(mesh)
    y.name = "y_eff"
    xdmf.write_function(y)
