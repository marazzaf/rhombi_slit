from scipy.optimize import minimize

#Minimization
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
res = minimize(fun, (2, 0))#, method='SLSQP')
assert res.success
print(res.x)
