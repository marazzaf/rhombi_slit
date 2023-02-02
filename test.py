from dolfin import *
import sys
import numpy as np

#mesh
L, H = 16, 16
N = 25
mesh = RectangleMesh(Point(0,0), Point(L,H), N, N, 'crossed')
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0) or near(x[1], H))
bnd = Bnd()
bnd.mark(boundaries, 1)

#Function space
V = FunctionSpace(mesh, "DG", 2)
print('Nb dof: %i' % V.dim())

#material parameters
alpha = -.9
beta = 0.21

#Compliance matrix
uu = Function(V, name='solution')
mu1 = cos(uu) - alpha*sin(uu)
mu1_p = -sin(uu) - alpha*cos(uu)
mu2 = cos(uu) + beta*sin(uu)
mu2_p = -sin(uu) + beta*cos(uu)
Gamma12 = -mu1_p / mu2
Gamma21 = mu2_p / mu1
Gamma = as_tensor(((-Gamma21, Constant(0)), (Constant(0), Gamma12)))

#Weak formulation
u = TrialFunction(V)
v = TestFunction(V)
pen = 10 #Make it a non-constant function?
h = CellDiameter(mesh)
hF = 0.5*(h('-')+h('+'))
n = FacetNormal(mesh)
sigma = dot(Gamma, grad(uu))
#Lhs
a = inner(sigma, grad(v)) * dx
a -= inner(dot(avg(sigma), n('+')), jump(v)) * dS
a += inner(dot(avg(dot(Gamma, grad(v))), n('+')), jump(uu)) * dS
a += pen/hF * inner(jump(uu), jump(v)) * dS

#Dirichlet BC
x = SpatialCoordinate(mesh)
xi = 0.5 * (1 - x[1]/H)  #-x[1]/H * 0.73
bcs = DirichletBC(V, xi, boundaries, 1, method='geometric')

#Rhs
L = Constant(0) * v * dx

#Linear solver to test
#solve(a == L, uu, bcs=bcs)                                             

#Newton solver
solve(a == 0, uu, bcs=bcs, solver_parameters={'newton_solver': {'maximum_iterations': 10}})

final = File('test.pvd')
final.write(uu)

poisson = File('test_poisson.pvd')
from ufl import sign
aux = project(sign(Gamma21*mu1**2/(Gamma12*mu2**2)), V)
poisson.write(aux)
