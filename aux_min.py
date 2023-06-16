from firedrake import *
import sys
sys.path.append('./experiments/')
from interpolate import BC
from firedrake.petsc import PETSc
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt

#Loading exp results
data = np.loadtxt('./experiments/aux_pull_new.txt', comments='#')

#Interpolate the exp results
exp_interp = LinearNDInterpolator(data[:,:2], data[:,2])

#Defining points to compare computation to experiment
top_right = np.array([[0.86, 0.86], [0.86, 0.76], [0.89, 0.60], [0.92, 0.30], [0.95,0.05]])
top_left = np.array([[-0.85, 0.90], [-0.88, 0.68], [-0.91, 0.45], [-0.95, 0.12]])
bottom_right = np.array([[0.93, -0.20], [0.90, -0.45], [0.88, -0.68], [0.87, -0.79], [0.85, -0.89]])
bottom_left = np.array([[-0.93, -0.26], [-0.90, -0.51], [-0.87, -0.75], [-0.86, -0.90]])
list_points_def = np.concatenate((top_right,bottom_right,top_left,bottom_left))

#Defining points where BC are optimized
H = 1.5
N = 11
aux = np.linspace(-H/2, H/2, N)
res = np.array([-H/2*np.ones_like(aux), aux]).T
list_points = np.concatenate((res,np.array([H/2*np.ones_like(aux), aux]).T))
#print(list_points.shape)

#Necessary for computation
mesh = Mesh('mesh_test.msh')

V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#material parameters
alpha = -.9
beta = 0.9

#Compliance matrix
xi = Function(V, name='xi')
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = diff(mu1, xi)
mu2_p = diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
B = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))

#Weak formulation of PDE
v = TestFunction(V)
a = inner(dot(B, grad(xi)), grad(v)) * dx

def min_BC(x): #values for the BC
    #BC coming from optimization.
    interp_BC = LinearNDInterpolator(list_points, x)

    #Using it as BC
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    vec_coord = X.dat.data_ro
    res_BC = Function(V)
    res = interp_BC(vec_coord[:,0],vec_coord[:,1])
    res_BC.dat.data[:] = res
    bcs = [DirichletBC(V, res_BC, 2)]

    #Check BC
    out_BC = File('bc.pvd')
    out_BC.write(res_BC)
    
    #Newton solver
    solve(a == 0, xi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})
    
    final = File('aux_pull_xi.pvd')
    final.write(xi)
    
    #poisson = File('aux_pull_poisson.pvd')
    #aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
    #poisson.write(aux)
    
    #Recovering global rotation
    #V = FunctionSpace(mesh, "CG", 1) #2
    u = TrialFunction(V)
    #v = TestFunction(V)
    aa = inner(grad(u), grad(v)) * dx
    #rhs
    Gamma = as_tensor(((Constant(0), Gamma12,), (Gamma21, Constant(0))))
    l = inner(dot(Gamma, grad(xi)), grad(v)) * dx
    
    gamma = Function(V, name='gamma')
    nullspace = VectorSpaceBasis(constant=True)
    #solve(a == l, gamma, nullspace=nullspace)
    A = assemble(aa)
    L = assemble(l)
    solve(A, gamma, L, nullspace=nullspace)
    
    #rotation = File('aux_pull_gamma.pvd')
    #rotation.write(gamma)
    
    #Recovering global disp
    A = as_tensor(((mu1, Constant(0)), (Constant(0), mu2)))
    R = as_tensor(((cos(gamma), -sin(gamma)), (sin(gamma), cos(gamma))))
    
    W = VectorFunctionSpace(mesh, 'CG', 1) #2
    uu = TrialFunction(W)
    vv = TestFunction(W)
    aaa = inner(grad(uu), grad(vv)) * dx
    ll = inner(dot(R, A), grad(vv))  * dx
    
    yeff = Function(W, name='yeff')
    aux = Function(W)
    solve(aaa == ll, yeff)
    
    #averaging
    pos_x = assemble(yeff[0] * dx) / assemble(1 * dx(mesh))
    pos_y = assemble(yeff[1] * dx) / assemble(1 * dx(mesh))
    yeff.interpolate(yeff - Constant((pos_x, pos_y)))
    disp = yeff.dat.data_ro - np.array([pos_x, pos_y])
    
    #disp = File('aux_pull_disp.pvd')
    #disp.write(yeff)

    disp_aux = File('aux_pull_disp_aux.pvd')
    x = SpatialCoordinate(mesh)
    yeff_aux = interpolate(yeff - as_vector((x[0], x[1])), W)
    disp_aux.write(yeff_aux)
    
    #Coords in deformed configuration
    def_coord = yeff.dat.data_ro
    
    #Constructing the interpolation
    res = LinearNDInterpolator(def_coord, xi.dat.data_ro, fill_value=10)
    #Where to get the points in def config...
    #Value of func at points in def configu
    list_xi_exp = exp_interp(list_points_def)
    #print(list_xi_exp)
    list_xi_comp = res(list_points_def)
    #print(list_xi_comp)

    ##Plot data
    #plt.scatter(def_coord[:,0], def_coord[:,1], c=xi.dat.data_ro, cmap='jet')
    #plt.colorbar()
    #plt.show()
    
    err = np.linalg.norm(list_xi_exp - list_xi_comp)
    print(err)

    return err
#    return (err, list_xi_exp - list_xi_comp)

#def jac(x):
#    return list_xi_exp - list_xi_comp

#Minimizing the BC
#initial = (0, 0.3, 0.74, 0.3, 0, 0, 0.3, 0.74, 0.3, 0)
initial = np.linspace(0, 0.74, int(N/2)+1)
initial = np.concatenate((initial, np.flip(initial)[1:]))
initial = np.concatenate((initial, initial))
#print(initial.shape)
bnds = np.tensordot(np.ones_like(initial), np.array([0, 0.74]), axes=0)
res_min = minimize(min_BC, initial, tol=1e-5, bounds=bnds) #, method='BFGS')
#res_min = minimize(min_BC, initial, tol=1e-3, jac=True, method='Newton-CG')
assert res_min.success

#Possible to add the jacobian to have faster convergence?
