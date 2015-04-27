#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

#a_np = np.random.rand(100).astype(np.float32)
#b_np = np.random.rand(100).astype(np.float32)

a_np = np.array([1,2,3],dtype=np.int64)
b_np = np.array([1,2,3],dtype=np.int64)

print 'a: ',a_np
print 'b: ',b_np
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(__global const int *a_g, __global const int *b_g, __global int *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
#prg.sum.set_scalar_arg_dtypes([np.int32, np.int32, np.int32])


prg.sum(queue, (6,), None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
#print(res_np - (a_np + b_np))
#print(np.linalg.norm(res_np - (a_np + b_np)))

print res_np