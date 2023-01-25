from firedrake import *
import sys

mesh = Mesh('mesh.msh')

V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v)) * dx

x = SpatialCoordinate(mesh)
F = Function(V)
F.interpolate(sin(x[0]*pi)*sin(2*x[1]*pi))
L = F*v*dx

bcs = [DirichletBC(V, Constant(2.0), 1)]

uu = Function(V)

# With the setup out of the way, we now demonstrate various ways of
# configuring the solver.  First, a direct solve with an assembled
# operator.::

solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                              "pc_type": "lu"})
out = File('out.pvd')
out.write(uu)

sys.exit()

# Finally, we demonstrate the use of a :class:`.AssembledPC`
# preconditioner.  This uses matrix-free actions but preconditions the
# Krylov iterations with an incomplete LU factorisation of the assembled
# operator.::

uu.assign(0)
solve(a == L, uu, bcs=bcs, solver_parameters={"mat_type": "matfree",
                                              "ksp_type": "cg",
                                              "ksp_monitor": None,

# To use the assembled matrix for the preconditioner we select a
# ``"python"`` type::

                                              "pc_type": "python",

# and set its type, by providing the name of the class constructor to
# PETSc.::

                                              "pc_python_type": "firedrake.AssembledPC",

# Finally, we set the preconditioner type for the assembled operator::

                                              "assembled_pc_type": "ilu"})

# This demo is available as a runnable python file `here
# <poisson.py>`__.
