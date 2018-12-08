# -*- coding:UTF-8 -*-
import numpy as np

def mode(b):
	b = b + 1
	print(b)

if __name__ == '__main__':
	a = 0
	b1 = np.array([[True], [False], [True], [False]])
	b2 = np.array([[True, False, True, False], 
				   [True, False, True, False]])
	
	for a in range(1, 10):
		print(a)
		mode(a)