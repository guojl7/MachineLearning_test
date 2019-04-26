# -*- coding:UTF-8 -*-
from theano import function, config, shared, tensor
import numpy
import time

if __name__ == '__main__':
	vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
	iters = 1000
	print("Resul")
	rng = numpy.random.RandomState(22)
	print("Resul1")
	x = shared(numpy.asarray(rng.rand(vlen), 'float32'))
	print("Resul2")
	f = function([], tensor.exp(x))
	print("Resul")
	print(f.maker.fgraph.toposort())
	t0 = time.time()
	for i in range(iters):
	    r = f()
	t1 = time.time()
	print("Looping %d times took %f seconds" % (iters, t1 - t0))
	print("Result is %s" % (r,))
	if numpy.any([isinstance(x.op, tensor.Elemwise) and ('Gpu' not in type(x.op).__name__) for x in f.maker.fgraph.toposort()]):
	    print('Used the cpu')
	else:
	    print('Used the gpu')
	
	def mode(b):
		b = b + 1
		print(b)