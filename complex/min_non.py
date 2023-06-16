import dolfinx
from mpi4py import MPI
import numpy as np
import sys
LL,H = 1.5,1.5
N = 20 #320 #160 #80
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[-LL/2,-H/2], [LL/2,H/2]], [N, N], diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
num_cells = mesh.topology.index_map(2).size_local
h = dolfinx.cpp.mesh.h(mesh, 2, range(num_cells))
h = h.max()
print('Size mesh: %.3e' % h)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
print('Nb dof: %i' % (V.dofmap.index_map.size_global * V.dofmap.index_map_bs))

from petsc4py import PETSc
assert np.dtype(PETSc.ScalarType).kind == 'c'

#Loading exp results
data = np.loadtxt('../experiments/non_pull_new.txt', comments='#')

#Interpolate the exp results
from scipy.interpolate import LinearNDInterpolator
exp_interp = LinearNDInterpolator(data[:,:2], data[:,2])

#Defining points to compare computation to experiment
#Update these
top_right = np.array([[0.86, 0.86], [0.86, 0.76], [0.89, 0.60], [0.92, 0.30], [0.95,0.05]])
top_left = np.array([[-0.85, 0.90], [-0.88, 0.68], [-0.91, 0.45], [-0.95, 0.12]])
bottom_right = np.array([[0.93, -0.20], [0.90, -0.45], [0.88, -0.68], [0.87, -0.79], [0.85, -0.89]])
bottom_left = np.array([[-0.93, -0.26], [-0.90, -0.51], [-0.87, -0.75], [-0.86, -0.90]])
list_points_def = np.concatenate((top_right,bottom_right,top_left,bottom_left))

#Defining points where BC are optimized
N = 6
aux = np.linspace(-H/2, H/2, N)
res = np.array([-H/2*np.ones_like(aux), aux]).T
list_points = np.concatenate((res,np.array([H/2*np.ones_like(aux), aux]).T))

import ufl
alpha = -.9
beta = 0.
xi = dolfinx.fem.Function(V, dtype=np.complex128)
mu1 = ufl.cos(xi) - alpha*ufl.sin(xi)
mu2 = ufl.cos(xi) + beta*ufl.sin(xi)
mu1_p = ufl.diff(mu1, xi)
mu2_p = ufl.diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
B = ufl.as_tensor(((-Gamma21, 0.), (0., Gamma12)))
delta = np.sqrt(h) #h #np.sqrt(h) #1e-2
B += ufl.as_tensor(((delta*1j, 0), (0, delta*1j)))

#Bilinear form
v = ufl.TestFunction(V)
a = ufl.inner(ufl.dot(B, ufl.grad(xi)), ufl.grad(v)) * ufl.dx

#Applying BC
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], LL))

#Recovering global rotation
u = ufl.TrialFunction(V)
aa = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
#rhs
Gamma = ufl.as_tensor(((0, Gamma12), (Gamma21, 0)))
L = ufl.inner(ufl.dot(Gamma, ufl.grad(xi)), ufl.grad(v)) * ufl.dx
gamma = dolfinx.fem.Function(V)

#Recovering global disp
A = ufl.as_tensor(((mu1, 0), (0, mu2)))
R = ufl.as_tensor(((ufl.cos(gamma), -ufl.sin(gamma)), (ufl.sin(gamma), ufl.cos(gamma))))

W = dolfinx.fem.VectorFunctionSpace(mesh, ('CG', 1))
uu = ufl.TrialFunction(W)
vv = ufl.TestFunction(W)
aaa = ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx
LL = ufl.inner(ufl.dot(R, A), ufl.grad(vv)) * ufl.dx
sys.exit()

#Procedure to minimize with respects to BC
def min_BC(x): #values for the BC
    #Interpolating the BC from optimization
    interp_BC = LinearNDInterpolator(list_points, x)

    #Using it as BC
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    vec_coord = X.dat.data_ro
    res_BC = Function(V)
    res = interp_BC(vec_coord[:,0],vec_coord[:,1])
    res_BC.dat.data[:] = res
    bcs = [DirichletBC(V, res_BC, 2)]
    u_bc.interpolate(xi_D)
    bc1 = dolfinx.fem.dirichletbc(u_bc, dofs_L)
    bc2 = dolfinx.fem.dirichletbc(u_bc, dofs_R)

    #Old boundary conditions
    val_max = 0.45 #0.8416
    val_min = 0 #0.0831
    x = ufl.SpatialCoordinate(mesh)
    aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
    aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
    xi_D = ufl.conditional(ufl.lt(x[1], H/2), aux2, aux1)
    xi_D = dolfinx.fem.Expression(xi_D, V.element.interpolation_points())
    u_bc = dolfinx.fem.Function(V, dtype=np.complex128)
    u_bc.interpolate(xi_D)
    
    #Nonlinear problem
    problem = dolfinx.fem.petsc.NonlinearProblem(a, xi, bcs=[bc1,bc2])
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

    #Assembling sys to get gamma
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(aa))
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
    # orthogonalize b with respect to the nullspace ensures that b does not contain any component in the nullspace
    nullspace.remove(b)
    
    # Finally we are able to solve our linear system
    solver.solve(b,gamma.vector)

    # Assemble system to get y_eff
    AA = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(aaa))
    AA.assemble()
    bb = dolfinx.fem.petsc.create_vector(dolfinx.fem.form(LL))
    with bb.localForm() as b_loc:
                b_loc.set(0)
    dolfinx.fem.petsc.assemble_vector(b,dolfinx.fem.form(LL))
    
    # Create Krylov solver
    solver = PETSc.KSP().create(AA.getComm())
    solver.setOperators(AA)
    
    # Create vector that spans the null space
    AA.setNullSpace(nullspace)
    # orthogonalize b with respect to the nullspace ensures that 
    # b does not contain any component in the nullspace
    nullspace.remove(b)
    
    # Finally we are able to solve our linear system ::
    y = dolfinx.fem.Function(W)
    solver.solve(bb,y.vector)
    
    #Writing output
    with dolfinx.io.XDMFFile(mesh.comm, "non_%i_disp.xdmf" % N, "w") as xdmf:
        xdmf.write_mesh(mesh)
        y.name = "y_eff"
        xdmf.write_function(y)

#Minimizing the BC
#initial = (0, 0.3, 0.74, 0.3, 0, 0, 0.3, 0.74, 0.3, 0)
initial = np.linspace(0, 0.74, int(N/2)+1)
initial = np.concatenate((initial, np.flip(initial)[1:]))
initial = np.concatenate((initial, initial))
#print(initial.shape)
bnds = np.tensordot(np.ones_like(initial), np.array([0, 0.74]), axes=0)
res_min = minimize(min_BC, initial, tol=1e-5, bounds=bnds)
assert res_min.success
