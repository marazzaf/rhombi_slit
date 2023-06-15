from scipy.optimize import minimize
import numpy as np

#Minimization
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
jac = lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2.5)])
#res = minimize(fun, (2, 0))
res = minimize(fun, (2, 0), jac=jac, method='CG')
assert res.success
print(res.x)
