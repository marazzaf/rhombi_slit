import dolfinx
from mpi4py import MPI
import numpy as np
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
u_r = dolfinx.fem.Function(V, dtype=np.float64) 
u_r.interpolate(lambda x: x[0])
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x:0.5*x[0]**2 + 1j*x[1]**2)
print(u_r.x.array.dtype)
print(u_c.x.array.dtype)

from petsc4py import PETSc
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1 - 2j))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

L2 = f * ufl.conj(v) * ufl.dx
print(L)
print(L2)

J = u_c**2 * ufl.dx
F = ufl.derivative(J, u_c, ufl.conj(v))
residual = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(F))
print(residual.array)

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_c, boundary_dofs)
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

import pyvista
pyvista.start_xvfb()
p_mesh = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim))
pyvista_cells, cell_types, geometry = dolfinx.plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u_real"] = uh.x.array.real
grid.point_data["u_imag"] = uh.x.array.imag
_ = grid.set_active_scalars("u_real")

p_real = pyvista.Plotter()
p_real.add_text("uh real", position="upper_edge", font_size=14, color="black")
p_real.add_mesh(grid, show_edges=True)
p_real.view_xy()
if not pyvista.OFF_SCREEN:
    p_real.show()

grid.set_active_scalars("u_imag")
p_imag = pyvista.Plotter()
p_imag.add_text("uh imag", position="upper_edge", font_size=14, color="black")
p_imag.add_mesh(grid, show_edges=True)
p_imag.view_xy()
if not pyvista.OFF_SCREEN:
    p_imag.show()
