import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import numpy as np
import pyopencl as cl

os.system("clear")

platform = cl.get_platforms()[0] 
device = platform.get_devices()[0]
ctx = cl.Context([device]) 
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

a_np = np.random.randint(0,10, size=(10,10)).flatten()
b_np = np.random.randint(0,10, size=(10,10)).flatten()
#a_np = np.random.rand(7000,8000).astype(np.float32)
#b_np = np.random.rand(7000,8000).astype(np.float32)

#print type(a_np)
print 'a_np\n',a_np
#print type(b_np)
print 'b_np\n',b_np

res_cl = np.empty_like(a_np)

a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=a_np.nbytes)

prg = cl.Program(ctx, """
__kernel void sum(__global  int *a, __global  int *b, __global int *c)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}""").build() 

#print a_np.shape

prg.sum(queue, (a_np.shape[0]*2,), None, a_g, b_g, res_g)
cl.enqueue_copy(queue, res_cl, res_g)


print "\nCL"
print res_cl
print "\nNP"
print a_np + b_np
