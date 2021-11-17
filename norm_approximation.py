import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve


def solve_linf(a, b):
	m = a.shape[0]
	n = a.shape[1]

	astack = matrix(np.vstack((a, -a)))
	tm_ones = matrix(-1*np.ones(2*m))
	Atilde = matrix([[astack], [tm_ones]])
	btilde = matrix(np.vstack((b, -1*b)))
	ctilde = np.zeros(n+1)
	ctilde[-1] = 1
	ctilde = matrix(ctilde)

	sol = solvers.lp(ctilde, Atilde, btilde)

	return sol['x'], sol['s']

def solve_lone(a, b):
	m = a.shape[0]
	n = a.shape[1]

	astack = matrix(np.vstack((a, -a)))
	tm_ones = matrix(np.vstack((-1*np.eye(m), -1*np.eye(m))))
	Atilde = matrix([[astack], [tm_ones]])
	btilde = matrix(np.vstack((b, -1*b)))

	czs = np.zeros(n)
	cones = np.ones(m)
	cstack = np.hstack((czs, cones))
	ctilde = matrix(cstack)

	sol = solvers.lp(ctilde, Atilde, btilde)

	return sol['x'], sol['s']


def solve_ltwo(a, b):
	S = np.matmul(a.T, a)
	q = np.matmul(a.T, b)
	cho_fac = cho_factor(S)
	x = cho_solve(cho_fac, q)
	residuals = b - np.matmul(a, x)
	return x, residuals
	