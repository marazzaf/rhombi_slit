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

#Interpolate the exp results?
exp_interp = LinearNDInterpolator(data[:,:2], data[:,2])

#test
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
res = minimize(fun, (2, 0))#, method='SLSQP')
assert res.success
#print(res.x)
#sys.exit()

mesh = Mesh('mesh_test.msh')
L, H = 1.6,1.6
#N = 5
#mesh = RectangleMesh(N,N,L,H,diagonal='crossed')

V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

##material parameters
alpha = -.9
beta = 0.9

#Complaince matrix
xi = Function(V, name='xi')
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = diff(mu1, xi)
mu2_p = diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))

#Weak formulation
v = TestFunction(V)
a = inner(dot(Gamma, grad(xi)), grad(v)) * dx

#old Dirichlet BC
val_max = 0.74
x = SpatialCoordinate(mesh)
aux1 = val_max * (1 - x[1]/H*2)
aux2 = val_max * (x[1]/H*2 + 1)
xi_D = conditional(lt(x[1], 0), aux2, aux1)
bcs = [DirichletBC(V, xi_D, 2)]

##BC coming from optimization.
#W = VectorFunctionSpace(mesh, V.ufl_element())
#X = interpolate(mesh.coordinates, W)
#vec_coord = X.dat.data_ro
#res_BC = Function(V)
#res = exp_interp(vec_coord[:,0],vec_coord[:,1])
#res_BC.dat.data[:] = res
#bcs = [DirichletBC(V, res_BC, 2)]

##Exit BC
#out_BC = File('bc.pvd')
#out_BC.write(res_BC)

#Newton solver
solve(a == 0, xi, bcs=bcs, solver_parameters={'snes_monitor': None, 'snes_max_it': 25})

final = File('aux_pull_xi.pvd')
final.write(xi)

#poisson = File('aux_pull_poisson.pvd')
#aux = interpolate(Gamma21*mu1**2/(Gamma12*mu2**2), V)
#poisson.write(aux)

#Recovering global rotation
V = FunctionSpace(mesh, "CG", 1) #2
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
#rhs
Gamma = as_tensor(((Constant(0), Gamma12,), (Gamma21, Constant(0))))
l = inner(dot(Gamma, grad(xi)), grad(v)) * dx

gamma = Function(V, name='gamma')
nullspace = VectorSpaceBasis(constant=True)
solve(a == l, gamma, nullspace=nullspace)

#rotation = File('rot.pvd')
#rotation.write(gamma)

#Recovering global disp
A = as_tensor(((mu1, Constant(0)), (Constant(0), mu2)))
R = as_tensor(((cos(gamma), -sin(gamma)), (sin(gamma), cos(gamma))))

W = VectorFunctionSpace(mesh, 'CG', 1) #2
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u), grad(v)) * dx
l = inner(dot(R, A), grad(v))  * dx

yeff = Function(W, name='yeff')
solve(a == l, yeff, nullspace=nullspace)

disp = File('aux_pull_disp.pvd')
disp.write(yeff)

#Constructing the interpolation to have xi_h on deformed mesh
W = VectorFunctionSpace(mesh, V.ufl_element())
X = interpolate(mesh.coordinates, W)
vec_coord = X.dat.data_ro
#x = vec_coord[:,0]
#y = vec_coord[:,1]
#print(vec_coord)
def_coord = vec_coord + yeff.dat.data_ro * 0.1 #coord in deformed mesh
#print(def_coord)

##Plot data
#plt.scatter(def_coord[:,0], def_coord[:,1], c=xi.dat.data_ro)
#plt.colorbar()
#plt.show()

#Constructing the interpolation
res = LinearNDInterpolator(def_coord, xi.dat.data_ro)

#Defining points to compare computation to experiment
list_pos = np.array([[-.7, 0], [.7, 0], [-.7,.7], [.7,.7], [-.7,-.7], [.7,-.7]])
list_xi_exp = exp_interp(list_pos)
list_xi_comp = res(list_pos)

err = np.linalg.norm(list_xi_exp - list_xi_comp)
print(err)
