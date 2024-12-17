from firedrake import *
from firedrake.output import VTKFile
import sys

#Mesh
N = 100 #320
L, H = 15, 15
mesh = RectangleMesh(N, N, L, H, diagonal='crossed', name='meshRef')

#Function Space
V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#Initial guess
v = TestFunction(V)
u = TrialFunction(V)

#Weak form
a = inner(grad(u), grad(v)) * dx
L = Constant(0) * v * dx

#Dirichlet BC
val_max = 0.74
val_min = 0
x = SpatialCoordinate(mesh)
aux1 = (val_max - val_min) * (2 - x[1]/H*2) + val_min
aux2 = (val_max - val_min)  * x[1]/H*2 + val_min
xi_D = conditional(lt(x[1], H/2), aux2, aux1)
bcs = [DirichletBC(V, xi_D, 1), DirichletBC(V, xi_D, 2)]

#Solving initial guess
xi = Function(V, name='xi')
solve(a == L, xi, bcs=bcs)

init = VTKFile('IG.pvd')
init.write(xi)

#material parameters
alpha = -.9
beta = 0.9

#Compliance matrix
#xi = variable(xi)
mu1 = cos(xi) - alpha*sin(xi)
mu2 = cos(xi) + beta*sin(xi)
mu1_p = diff(mu1, xi)
mu2_p = diff(mu2, xi)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))
DGamma = diff(Gamma, xi)

#Xi = Function(V)
#test = replace(DGamma, {xi: Xi}) #real(xi)})
#sys.exit()

#Weak formulation
#epsilon = .5
#Gamma += epsilon * Constant(((1j, 0), (0, 1j))) #Adding a complex dissipation
a = inner(dot(Gamma, grad(u)), grad(v)) * dx + inner(u * dot(DGamma, grad(xi)), grad(v)) * dx

#RHS
L = -inner(dot(Gamma, grad(xi)), grad(v)) * dx

#Homogeneous Dirichlet BC for increment
bcs = [DirichletBC(V, Constant(0), 1), DirichletBC(V, Constant(0), 2)]

# Newton iteration
tol = 1e-5
maxiter = 30
d_xi = Function(V, name='increment')
#Xi = Function(V)
#Gamma = replace(Gamma, {xi:Xi})
#DGamma = replace(Gamma, {xi:Xi})
for iter in range(maxiter):
    # compute the Newton increment by solving the linearized problem
    solve(a == L, d_xi, bcs=bcs) 
    xi.assign(xi + d_xi) # update the solution
    eps = norm(d_xi, 'h1') #check increment size as convergence test
    print('iteration{:3d}  H1 seminorm of delta: {:10.2e}'.format(iter+1, eps))
    if eps < tol:
      break

#Writing the result
final = VTKFile('test.pvd')
final.write(xi)
