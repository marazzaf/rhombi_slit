import dolfinx
from mpi4py import MPI
import numpy as np
LL,H = 1,1
N = 6 #10 #20 #40 #80 #160 #320
infile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./mesh_%i.xdmf" % N, "r")
mesh = infile.read_mesh(name="Grid")
infile.close()
#mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[-LL,-H], [LL,H]], [N, N], diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
num_cells = mesh.topology.index_map(2).size_local
h = dolfinx.cpp.mesh.h(mesh, 2, range(num_cells))
h = h.max()
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

import ufl
aux = dolfinx.fem.Function(V, dtype=np.complex128)
print('nb dof: %i' % aux.vector.size)

#matrix
delta = 1e-3*h  #1e-3*h #1e-3*np.sqrt(h) #1e-3
#aux.interpolate(lambda x: x[0] * (1 + delta*np.sign(x[0]) * 1j))
#test
aux.interpolate(lambda x: x[0] + delta * 1j)
Gamma = ufl.as_tensor(((aux, 0.), (0., 1 + delta * 1j)))


#Bilinear form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.dot(Gamma, ufl.grad(u)), ufl.grad(v)) * ufl.dx

#linear form
truc = dolfinx.fem.Function(V, dtype=np.complex128)
truc.interpolate(lambda x: -2*x[0] * np.sign(x[0]) +0*1j)

L = ufl.inner(truc, v) * ufl.dx

#Boundary conditions
u_bc = dolfinx.fem.Function(V, dtype=np.complex128)
xi = lambda x: 0.5 * x[0]*x[0] * np.sign(x[0]) + 0*1j
u_bc.interpolate(xi)
mesh.topology.create_connectivity(1, 2)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_bc, boundary_dofs)

#Linear problem
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

uR = dolfinx.fem.Function(V, dtype=np.complex128)
uR.x.array[:] = uh.x.array.real
with dolfinx.io.XDMFFile(mesh.comm, "conv_%i.xdmf" % N, "w") as xdmf:
    xdmf.write_mesh(mesh)
    uh.name = "Approx"
    xdmf.write_function(uh)

with dolfinx.io.XDMFFile(mesh.comm, "ref_%i.xdmf" % N, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xi.name = "Ref"
    xdmf.write_function(u_bc)

def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = dolfinx.fem.FunctionSpace(mesh, (family, degree+ degree_raise))
    # Interpolate approximate solution
    u_W = dolfinx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)
    
    # Compute the error in the higher order function space
    e_W = dolfinx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    
    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

print('L2 error: %.3e' % error_L2(uR, xi).real)
