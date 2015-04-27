#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
from pyopencl import array
from pyopencl import clrandom
from pyopencl import clmath
import time
#import gc
import os
import sys
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

class OpenClnaMao():
	def __init__(self):#Set openCl platform and device that will be use
		deviceID = 0 #OpenCl device
		platformID = 0 #OpenCl platform
		self.dev = cl.get_platforms()[platformID].get_devices()[deviceID]
		self.ctx = cl.Context([self.dev])
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.mf = cl.mem_flags

	def add_nm(self,tam,a_np,b_np):
		res_cl = np.empty_like(a_np)

		a_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a_np.flatten())
		b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_np.flatten())
		res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, size=a_np.nbytes)

		prg = cl.Program(self.ctx, """
		__kernel void sum(__global  int *a, __global  int *b, __global int *c)
		{
		  int i = get_global_id(0);
		  c[i] = a[i] + b[i];
		}""").build() 

		#print a_np.shape

		prg.sum(self.queue, (a_np.shape[0]*a_np.shape[1],4), None, a_g, b_g, res_g)
		cl.enqueue_copy(self.queue, res_cl, res_g)
		return res_cl#.reshape(tam[0],tam[1])

	def sub_nm(self,tam,a_np,b_np):
		res_cl = np.empty_like(a_np)

		a_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a_np.flatten())
		b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_np.flatten())
		res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, size=a_np.nbytes)

		prg = cl.Program(self.ctx, """
		__kernel void sum(__global  int *a, __global  int *b, __global int *c)
		{
		  int i = get_global_id(0);
		  c[i] = a[i] - b[i];
		}""").build() 

		#print a_np.shape

		prg.sum(self.queue, (a_np.shape[0]*2,), None, a_g, b_g, res_g)
		cl.enqueue_copy(self.queue, res_cl, res_g)
		return res_cl.reshape(tam[0],tam[1])

	def div_nm(self,tam,a_np,b_np):
		res_cl = np.empty_like(a_np)

		a_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a_np.flatten())
		b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_np.flatten())
		res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, size=a_np.nbytes)

		prg = cl.Program(self.ctx, """
		__kernel void sum(__global  int *a, __global  int *b, __global int *c)
		{
		  int i = get_global_id(0);
		  c[i] = a[i] / b[i];
		}""").build() 

		#print a_np.shape

		prg.sum(self.queue, (a_np.shape[0]*2,), None, a_g, b_g, res_g)
		cl.enqueue_copy(self.queue, res_cl, res_g)
		return res_cl.reshape(tam[0],tam[1])

	def mul_nm(self,tam,a_np,b_np):
		res_cl = np.empty_like(a_np)

		a_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=a_np.flatten())
		b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_np.flatten())
		res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, size=a_np.nbytes)

		prg = cl.Program(self.ctx, """
		__kernel void sum(__global  int *a, __global  int *b, __global int *c)
		{
		  int i = get_global_id(0);
		  c[i] = a[i] * b[i];
		}""").build() 

		#print a_np.shape

		prg.sum(self.queue, (a_np.shape[0]*2,), None, a_g, b_g, res_g)
		cl.enqueue_copy(self.queue, res_cl, res_g)
		return res_cl.reshape(tam[0],tam[1])

class OpenClFun():
	def __init__(self):#Set openCl platform and device that will be use
		deviceID = 0 #OpenCl device
		platformID = 0 #OpenCl platform
		self.dev = cl.get_platforms()[platformID].get_devices()[deviceID]
		self.ctx = cl.Context([self.dev])
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

	def GetReadyOpen(self,*args):
		args = list(args)
		for i in xrange(0,len(args)):
			args[i] = self.np_into_cl(args[i])

		if len(args)>1:
			res = args[0]
			for i in xrange(1,len(args)):
				res = res + args[i]
		
		res = self.cl_into_np(res)
		print "Pyopencl Ready"

	def np_into_cl(self,nparray):
		#numpy.ndarray into a openCl.array
		return cl.array.to_device(self.queue,nparray, allocator=None, async=False)

	def cl_into_np(self,clarray):
		#openCl.array into a numpy.ndarray
		return cl.array.Array.get(clarray,self.queue,async=False)
		
	def cl_add(self,*args):
		args = list(args)

		#numpy.ndarray into opencl.array	
		s = time.time()
		r = np.empty_like(args[0])
		for i in xrange(0,len(args)):
			args[i] = self.np_into_cl(args[i])
		print "-NP to CL: ",time.time() - s

		s = time.time()
		if len(args)>1:
			res = args[0]
			for i in xrange(1,len(args)):
				res = res + args[i]
		print "-OCL calc: ",time.time() - s
		
		s = time.time()
		res = self.cl_into_np(res)
		print "-CL to NP: ",time.time() - s

		return res

	def cl_sub(self,*args):
		args = list(args)

		#numpy.ndarray into opencl.array	
		s = time.time()
		for i in xrange(0,len(args)):
			args[i] = self.np_into_cl(args[i])
		print "NP to CL: ",time.time() - s

		s = time.time()
		if len(args)>1:
			res = args[0]
			for i in xrange(1,len(args)):
				res = res - args[i]
		print "OCL calc: ",time.time() - s
		
		s = time.time()
		#res = self.cl_into_np(res)
		res = res.get()
		print "CL to NP: ",time.time() - s
		
		return res

	def cl_mul(self,*args):
		args = list(args)

		#numpy.ndarray into opencl.array	
		s = time.time()
		for i in xrange(0,len(args)):
			args[i] = self.np_into_cl(args[i])
		print "NP to CL: ",time.time() - s

		s = time.time()
		if len(args)>1:
			res = args[0]
			for i in xrange(1,len(args)):
				res = res * args[i]
		print "OCL calc: ",time.time() - s
		
		s = time.time()
		res = self.cl_into_np(res)
		print "CL to NP: ",time.time() - s
		
		return res

	def cl_div(self,*args):
		args = list(args)

		#numpy.ndarray into opencl.array	
		s = time.time()
		for i in xrange(0,len(args)):
			args[i] = self.np_into_cl(args[i])
		print "NP to CL: ",time.time() - s

		s = time.time()
		if len(args)>1:
			res = args[0]
			for i in xrange(1,len(args)):
				res = res / args[i]
		print "OCL calc: ",time.time() - s
		
		s = time.time()
		res = self.cl_into_np(res)
		print "CL to NP: ",time.time() - s
		
		return res

	def cl_pow(self,array,exp):

		s = time.time()
		array = self.np_into_cl(array)
		print "NP to CL: ",time.time() - s

		tmp = array
		s = time.time()
		for i in xrange(exp):
			array = array*tmp
		print "OCL calc: ",time.time() - s

		s = time.time()
		array = self.cl_into_np(array)
		print "CL to NP: ",time.time() - s

		return array

	def cl_log(self,array):
		if type(array) == 'np.ndarray':	
			array = self.np_into_cl(array)
		
		res = array._new_like_me()

		if type(res) == 'np.ndarray':
			return self.cl_into_np(cl.clmath.log(res,queue=None))
		else:
			return res

#log raiz
class CompareNpCl():
	#@profile
	def __init__(self):
		tam = np.array((10,10))
		np_a = np.random.randint(-32000,32000, size=tam)
		np_b = np.random.randint(-32000,32000, size=tam)
		oclf.GetReadyOpen(np_a,np_b)
		tam = None
		np_a = None
		np_b = None

	def comp_add(self,tam,np_a,np_b):
		print "Comp add: ",tam[0]," x ",tam[1]
		
		t = time.time()
		for i in xrange(1):
			r_np = np_a + np_b
		print "NP tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_cl = oclf.cl_add(np_a,np_b)
		print "CL tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_clnm = oclnm.add_nm(tam,np_a,np_b)
		print "CLNM tempo: ",time.time() - t


		#r_nm = oclnm.add_nm(tam,np_a,np_b)
		np_a = None
		np_b = None
		assert r_np.all() == r_cl.all()
		assert r_np.all() == r_clnm.all()
		r_np = None
		r_cl = None

	def comp_sub(self,tam,np_a,np_b):
		print "Comp sub: ",tam[0]," x ",tam[1]
		
		t = time.time()
		for i in xrange(1):
			r_np = np_a - np_b
		print "NP tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_cl = oclf.cl_sub(np_a,np_b)
		print "CL tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_clnm = oclnm.sub_nm(tam,np_a,np_b)
		print "CLNM tempo: ",time.time() - t

		np_a = None
		np_b = None
		assert r_np.all() == r_cl.all()
		r_np = None
		r_cl = None

	def comp_mul(self,tam,np_a,np_b):
		print "Comp mul: ",tam[0]," x ",tam[1]
		
		t = time.time()
		for i in xrange(1):
			r_np = np_a * np_b
		print "NP tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_cl = oclf.cl_mul(np_a,np_b)
		print "CL tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_clnm = oclnm.mul_nm(tam,np_a,np_b)
		print "CLNM tempo: ",time.time() - t

		np_a = None
		np_b = None
		assert r_np.all() == r_cl.all()
		r_np = None
		r_cl = None

	def comp_div(self,tam,np_a,np_b):
		print "Comp div: ",tam[0]," x ",tam[1]
		
		t = time.time()
		for i in xrange(1):
			r_np = np_a / np_b
		print "NP tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_cl = oclf.cl_div(np_a,np_b)
		print "CL tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_clnm = oclnm.div_nm(tam,np_a,np_b)
		print "CLNM tempo: ",time.time() - t

		np_a = None
		np_b = None
		r_np = None
		r_cl = None

	def comp_pow(self,tam,np_a):
		print "Comp pow: ",tam[0]," x ",tam[1]
		
		t = time.time()
		for i in xrange(1):
			r_np = np_a**3
		print "NP tempo: ",time.time() - t

		t = time.time()
		for i in xrange(1):
			r_cl = oclf.cl_pow(np_a,3)
		print "CL tempo: ",time.time() - t

		np_a = None
		np_b = None
		assert r_np.all() == r_cl.all()
		r_np = None
		r_cl = None

#@profile
def main():
	tam = np.array((8000,8000))
	#np_a = np.random.randint(-32000,32000, size=tam).astype(np.float32)
	#np_b = np.random.randint(-32000,32000, size=tam).astype(np.float32)

	np_a = np.random.rand(10000,10000).astype(np.float32)
	np_b = np.random.rand(10000,10000).astype(np.float32)

	oclf = OpenClFun()
	oclnm = OpenClnaMao()
	global oclf
	global oclnm
	cpnpcl = CompareNpCl()

	
	print "\n---ADD---"
	cpnpcl.comp_add(tam,np_a,np_b)

	'''
	print "\n---SUB---"
	cpnpcl.comp_sub(tam,np_a,np_b)

	print "\n---MUL---"
	cpnpcl.comp_mul(tam,np_a,np_b)

	print "\n---DIV---"
	cpnpcl.comp_div(tam,np_a,np_b)
	
	print "\n---POW---"
	cpnpcl.comp_pow(tam,np_a)
	'''
	tam = None
	np_a = None
	np_b = None

if __name__ == '__main__':
	os.system("clear")
	for i in xrange(1):
		print '\ntry: ',i+1
		main()
