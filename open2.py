import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import numpy as np
import pyopencl as cl
import time


def main():
	platform = cl.get_platforms()[0] 
	device = platform.get_devices()[0]
	ctx = cl.Context([device]) 
	queue = cl.CommandQueue(ctx)
	mf = cl.mem_flags

	#a_np = np.random.randint(0,10, size=(7000,8000)).flatten()
	#b_np = np.random.randint(0,10, size=(7000,8000)).flatten()

	a_np = np.random.rand(10000,10000).astype(np.float32)
	b_np = np.random.rand(10000,10000).astype(np.float32)

	res_cl = np.empty_like(a_np)

	##############OpenCL###################
	print "\nCL instancias"
	inicio = time.time()

	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np.flatten())
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np.flatten())
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=a_np.nbytes)

	prg = cl.Program(ctx, """
	__kernel void sum(__global  float *a, __global  float *b, __global float *c)
	{
	  int i = get_global_id(0);
	  c[i] = a[i] + (b[i]*2) + (b[i]/2) + b[i];
	}""").build() 

	fim = time.time()
	print (fim - inicio),' segundos.'

	print "\nCL calculo"
	inicio = time.time()

	prg.sum(queue, (a_np.shape[0]*a_np.shape[1],),(1,), None, a_g, b_g, res_g)
	cl.enqueue_copy(queue, res_cl, res_g)

	fim = time.time()

	print (fim - inicio),' segundos.'


	##############NumPy###################
	print "\nNP"
	inicio = time.time()
	r_np = a_np + (b_np*2) + (b_np/2) + b_np
	fim = time.time()
	print (fim - inicio), ' segundos.'
	
	###Confirma###
	print "\nCL\n", res_cl
	print "\nNP\n", r_np

	assert res_cl.all() == r_np.all()

if __name__ == '__main__':
	os.system("clear")
	for x in xrange(0,1):
		main()