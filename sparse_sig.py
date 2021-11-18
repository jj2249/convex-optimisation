import numpy as np
from scipy.linalg import cho_factor, cho_solve

def grad_phi(A, b, l, t, x, u):
	"""
	Gradient of central path for q3
	"""
	n = x.shape[0]

	gradx1 = 2.*t*np.matmul(A.T, np.matmul(A, x)-b.flatten()) 
	gradx2 = np.divide(2.*x, np.square(u)-np.square(x))
	gradx = np.add(gradx1, gradx2)

	gradu1 = t*l * np.ones(n)
	gradu2 = np.divide(-2.*u, np.square(u)-np.square(x))
	gradu = np.add(gradu1, gradu2)

	return np.hstack((gradx, gradu))


def hessian_phi(A, b, t, x, u):
	"""
	Hessian of central path for q3
	"""
	gradxx1 = 2.*t*np.matmul(A.T, A)
	gradxx2 = np.diag(np.divide(1., np.square(u+x)) + np.divide(1., np.square(u-x)))
	gradxx = np.add(gradxx1, gradxx2)

	graduu = np.diag(np.divide(1., np.square(u+x)) + np.divide(1., np.square(u-x)))

	gradux = np.diag(np.divide(1., np.square(u+x)) - np.divide(1., np.square(u-x)))

	return np.block([[gradxx, gradux], [gradux, graduu]])


def central_phi(A, b, l, t, x, u):
	"""
	Central path for q3
	"""
	term1 = np.linalg.norm(np.matmul(A, x) - b)
	term2 = t*l*np.sum(u)
	term3 = -1.*np.sum(np.log(u+x))-1.*np.sum(np.log(u-x))

	return term1 + term2 + term3


def newton_step(A, b, l, t, coords):
	"""
	Take a newton step
	"""
	x = coords[:256]
	u = coords[256:]

	grad = grad_phi(A, b, l, t, x, u)
	hessian = hessian_phi(A, b, t, x, u)

	# avoid inverting the 512x512 hessian -- xk+1 = xk - h where h =H^-1g
	c = cho_factor(hessian)
	step = cho_solve(c, grad)
	coord_new = coords - step
	# vec = np.matmul(hessian, coords) - grad
	# c = cho_factor(hessian)
	# xnew = cho_solve(c, vec)
	return coord_new

def minimum_energy(A, b):
	"""
	Minimum Energy Reconstruction
	"""
	return np.linalg.lstsq(A, b)[0]