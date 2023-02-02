from dolfin import *
import sys
import numpy as np
from ufl import sign

#mesh
L, H = 16, 16
N = 50
mesh = RectangleMesh(Point(0,0), Point(L,H), N, N, 'left')
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0) or near(x[1], H)) #and (near(x[0], 0) or near(x[0], L)) #and (near(x[1], 0) or near(x[1], H))
bnd = Bnd()
bnd.mark(boundaries, 1)

#Function space
V = FunctionSpace(mesh, "CG", 2)
print('Nb dof: %i' % V.dim())

#material parameters
alpha = -.9
beta = 0.21 #0 #0.54 #0.21

#Compliance matrix
uu = Function(V, name='solution')
mu1 = cos(uu) - alpha*sin(uu)
mu1_p = -sin(uu) - alpha*cos(uu)
mu2 = cos(uu) + beta*sin(uu)
mu2_p = -sin(uu) + beta*cos(uu)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))
poisson = Gamma21*mu1**2/(Gamma12*mu2**2)

#Weak formulation
u = TrialFunction(V)
v = TestFunction(V)
pen = 10 * (sign(poisson)('-') + sign(poisson)('+')) #Make it a non-constant function?
h = CellDiameter(mesh)
hF = 0.5*(h('-')+h('+'))
n = FacetNormal(mesh)
sigma = dot(Gamma, grad(uu))
#Lhs
a = inner(sigma, grad(v)) * dx
a = inner(sigma, dot(Gamma, grad(v))) * dx
#a -= inner(dot(avg(sigma), n('+')), jump(v)) * dS
#a += inner(dot(avg(dot(Gamma, grad(v))), n('+')), jump(uu)) * dS
#a += pen/hF * inner(jump(uu), jump(v)) * dS

#Dirichlet BC
x = SpatialCoordinate(mesh)
xi = 0.5 * (1 - x[1]/H)
bcs = DirichletBC(V, xi, boundaries, 1, method='geometric')

#Rhs
#L = Constant(0) * v * dx
#Linear solver to test
#solve(a == L, uu, bcs=bcs)                                             

#Newton solver
#solve(a == 0, uu, bcs=bcs)
try:
    solve(a == 0, uu, bcs=bcs, solver_parameters={"nonlinear_solver": "snes", 'snes_solver': {'maximum_iterations': 10}}) #, 'relative_tolerance': 1e-6}})#, 'linear_solver': 'gmres', 'preconditioner': 'ilu'}})
except RuntimeError:
    pass
    
final = File('test.pvd')
final.write(uu)

Poisson = File('test_poisson.pvd')
aux = project(poisson, V)
Poisson.write(aux)
