import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import norm


def central_path(x, c, a, b, t):
	x = matrix(x)
	term1 = t*np.dot(c.T, x)
	# term2 = 0
	# for i in range(b.size[0]):
	# 		if b[i]-np.dot(a[i,:], x) < 0:
	# 			return np.inf
	# 		else:
	# 			term2 += np.log(b[i]-np.dot(a[i,:], x))
	# print('b: '+str(b.size))
	# print('a: '+str(a.size))
	# print('x: '+str(x.size))
	term2 = np.sum(np.log(b - (a*x)), axis=0)
	# print('term2: '+str(term2[0]))
	if np.sum(np.isnan(term2))>0:
		# print('infinite')
		return matrix(np.inf*np.ones(1))
	else:
		return term1 - term2


def central_path_grad(x, c, a, b, t):
	term1 = np.array(t*c)
	# term2 = np.zeros((1, c.size[0]))
	# for i in range(b.size[0]):
	# 	term2 += a[i,:]/(b[i]-np.dot(a[i,:], x))
	term2 = np.sum(a/np.array(b - a*x), axis=0)
	total = term1.flatten() + term2
	return matrix(total)


def build_l1_tildes(a, b):
	m = a.shape[0]
	n = a.shape[1]
	at = np.block([[a, -1*np.eye(m)],[-a, -1*np.eye(m)]])
	# at = np.hstack((astack, tm_ones))
	bt = np.block([[b], [-1*b]])
	return matrix(at), matrix(bt)


def find_bfs(a, b, m, n):
	aones = matrix(-1*np.ones(2*m))
	ai = matrix(np.hstack((matrix(a), aones)))
	bi = matrix(b)
	c = np.zeros(n+m+1)
	c[-1] = 1
	ci = matrix(c)

	sol = solvers.lp(ci, ai, bi)
	bfs = sol['x'][:-1]

	return bfs

def backtrack(x, data, func, grad, alpha, beta, cvec, u):
	x = matrix(x)
	t = 1
	current_func = func(x, cvec, data[0], data[1], u)
	gradient = grad(x, cvec, data[0], data[1], u)
	step = -1*alpha*(np.linalg.norm(gradient)**2)
	new_func = func((x-t*gradient), cvec, data[0], data[1], u)
	# print('numbers'+str(current_func)+str(new_func)+str(step))

	while new_func[0] > (current_func[0]+t*step):
		# print('comparison: '+str(new_func[0] - current_func[0] - t*step))
		t *= beta
		new_func = func((x-t*gradient), cvec, data[0], data[1], u)
	return t


def first_order_method(data, func, grad, cvec, epsilon, m, n, u=1.):
	x = find_bfs(data[0], data[1], m, n)
	# x = np.zeros(m+n)
	# print('func shape: ' + str(func(x, cvec, data[0], data[1], u).size))
	# print('grad shape: ' + str(grad(x, cvec, data[0], data[1], u).size))
	iteration=0
	errors = []
	while np.linalg.norm(grad(x, cvec, data[0], data[1], u)) > epsilon:
		# print(iteration, func(x, cvec, data[0], data[1], u))
		step_dir = -1*grad(x, cvec, data[0], data[1], u)
		step_len = backtrack(x, data, func, grad, 0.2, 0.4, cvec, u)
		# print('step len: '+str(step_len))
		x += (step_len * step_dir)
		iteration +=1
		errors.append(func(x, cvec, data[0], data[1], u)[0][0])
	return x, func(x, cvec, data[0], data[1], u)[0], errors
