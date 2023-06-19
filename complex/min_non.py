import dolfinx
from mpi4py import MPI
import numpy as np
import sys
LL,H = 1.55,1.55
NN = 200 #320 #160 #80
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[-LL/2,-H/2], [LL/2,H/2]], [NN, NN], diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
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
top_right = np.array([[0.85, 0.69], [0.86, 0.50], [0.90, 0.13]])
top_left = np.array([[-0.82, 0.68], [-0.85, 0.40], [-0.88, 0.12]])
bottom_right = np.array([[0.89, -0.14], [0.85, -0.51], [0.83, -0.69]])
bottom_left = np.array([[-0.90, -0.05], [-0.87, -0.24], [-0.84, -0.43], [-0.80, -0.71]])
list_points_def = np.concatenate((top_right,bottom_right,top_left,bottom_left))

#Defining points where BC are optimized
N = 11
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
dofs_L = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], -LL/2))
dofs_R = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], LL/2))

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

#Procedure to minimize with respects to BC
def min_BC(x): #values for the BC
    #Interpolating the BC from optimization
    interp_BC = LinearNDInterpolator(list_points, x)

    #Using it as BC
    vec_coord = mesh.geometry.x[:,:2]
    res_bc = dolfinx.fem.Function(V)
    res = interp_BC(vec_coord[:,0],vec_coord[:,1])
    res_bc.vector[:] = res
    bc1 = dolfinx.fem.dirichletbc(res_bc, dofs_L)
    bc2 = dolfinx.fem.dirichletbc(res_bc, dofs_R)
    
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
    
    with dolfinx.io.XDMFFile(mesh.comm, "non_%i_xi.xdmf" % NN, "w") as xdmf:
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
    dolfinx.fem.petsc.assemble_vector(bb,dolfinx.fem.form(LL))
    
    # Create Krylov solver
    solver = PETSc.KSP().create(AA.getComm())
    solver.setOperators(AA)
    
    # Create vector that spans the null space
    nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
    AA.setNullSpace(nullspace)
    nullspace.remove(bb)
    
    # Finally we are able to solve our linear system ::
    y = dolfinx.fem.Function(W)
    solver.solve(bb,y.vector)
    
    #Writing output for defor
    x = ufl.SpatialCoordinate(mesh)
    yeff_aux = dolfinx.fem.Function(W)
    yeff_aux.interpolate(y)
    yeff_aux.vector[:] -= vec_coord.flatten()
    with dolfinx.io.XDMFFile(mesh.comm, "non_%i_disp_aux.xdmf" % NN, "w") as xdmf:
        xdmf.write_mesh(mesh)
        yeff_aux.name = "y_eff"
        xdmf.write_function(yeff_aux)

    #Writing output
    with dolfinx.io.XDMFFile(mesh.comm, "non_%i_disp.xdmf" % NN, "w") as xdmf:
        xdmf.write_mesh(mesh)
        y.name = "y_eff"
        xdmf.write_function(y)

    #Coords in deformed configuration
    def_coord = y.vector.array
    def_coord = def_coord.reshape(def_coord.size //2, 2)
    
    #Constructing the interpolation
    res = LinearNDInterpolator(def_coord, xi.vector.array, fill_value=10)
    ##Value of func at points in def configu
    #list_xi_exp = exp_interp(list_points_def)
    #print(list_xi_exp)
    #list_xi_comp = res(list_points_def)
    #print(list_xi_comp)
    #err = np.linalg.norm(list_xi_exp - list_xi_comp)
    #print(err)

    #Test with all values
    list_xi_comp = res(data[:,:2])
    #print(list_xi_comp)
    sys.exit()


    err = np.linalg.norm(list_xi_comp - data[:,2])
    print(err)
    return err

#Minimizing the BC
#initial = (0, 0.3, 0.74, 0.3, 0, 0, 0.3, 0.74, 0.3, 0)
from scipy.optimize import minimize
initial = np.linspace(0, 0.45, int(N/2)+1)
initial = np.concatenate((initial, np.flip(initial)[1:]))
initial = np.concatenate((initial, initial))
#print(initial.shape)
bnds = np.tensordot(np.ones_like(initial), np.array([0, 0.45]), axes=0)
res_min = minimize(min_BC, initial, tol=1e-5, bounds=bnds)
assert res_min.success

