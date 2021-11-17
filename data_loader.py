import numpy as np
import scipy.io as io
from sys import argv

def mat_to_np(file):
	with open(file+'.mat', 'rb') as f:
		data = io.loadmat(f)
		f.close()
	with open(file+'.npy', 'wb') as f:
		np.save(f, data)
		f.close()


def load_npy(number):
	with open("./q1/A{}.npy".format(number), 'rb') as f:
		data = np.load(f, allow_pickle=True).item() # 0 dimensional array of object dtype
		a = data['A{}'.format(number)]
		f.close()
	with open("./q1/b{}.npy".format(number), 'rb') as f:
		data = np.load(f, allow_pickle=True).item() # 0 dimensional array of object dtype
		b = data['b{}'.format(number)]
		f.close()
	
	return a, b

def load_signal():
	with open("./q3/A.npy", 'rb') as f:
		data = np.load(f, allow_pickle=True).item()
		A = data['A']
		f.close()
	with open("./q3/x0.npy", 'rb') as f:
		data = np.load(f, allow_pickle=True).item()
		x0 = data['x0']
		f.close()

	return A, x0

	
if __name__ == "__main__":
	mat_to_np(argv[1])