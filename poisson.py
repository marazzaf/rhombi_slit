from firedrake import *
import sys

mesh = Mesh('mesh.msh')

V = FunctionSpace(mesh, "CG", 1)

x = SpatialCoordinate(mesh)
sig_1 = Constant(1)
sig_2 = Constant(-1.5)
sigma = conditional(lt(x[0], Constant(0)), sig_1, sig_2)
gamma = 1e1
aux = 1 + complex(0,gamma) * sign(sigma)

u = TrialFunction(V)
v = TestFunction(V)

a = aux * sigma * inner(grad(u), grad(v)) * dx

xi_1 = ((x[0]+1)**2 - (2*sig_1+sig_2)*(x[0]+1) / (sig_1+sig_2)) * sin(pi*x[1])
xi_2 = sig_1 * (x[0] - 1) * sin(pi*x[1]) / (sig_1+sig_2)
xi = conditional(lt(x[0], Constant(0)), xi_1, xi_2)

bcs = [DirichletBC(V, xi, 1)]

L = -div(sigma*grad(xi)) * v * dx

uu = Function(V, name='solution')

# With the setup out of the way, we now demonstrate various ways of
# configuring the solver.  First, a direct solve with an assembled
# operator.::

solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                              "pc_type": "lu"})
sys.exit()

out = File('out.pvd')
out.write(uu)

ref = File('ref.pvd')
vv = Function(V, name='ref')
vv.interpolate(xi)
ref.write(vv)

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
