try:
	import numpy as np
	has_numpy = True
except ImportError:
	import math
	has_numpy = False
try:
	import scipy.constants
	has_scipy = True
except ImportError:
	has_scipy = False
import operator as op
from .similar import sim, nsim, gsim, lsim

def equation_extend(core):
	def product(*args):
		if len(args) == 1 and has_numpy:
			return np.prod(args[0])
		else:
			return reduce(op.mul,args,1)

	def sumargs(*args):
		if len(args) == 1:
			return sum(args[0])
		else:
			return sum(args)

	core.addOp('+',"({0:s} + {1:s})","\\left({0:s} + {1:s}\\right)",False,3,op.add)
	core.addOp('-',"({0:s} - {1:s})","\\left({0:s} - {1:s}\\right)",False,3,op.sub)
	core.addOp('*',"({0:s} * {1:s})","\\left({0:s} \\times {1:s}\\right)",False,2,op.mul)
	core.addOp('/',"({0:s} / {1:s})","\\frac{{{0:s}}}{{{1:s}}}",False,2,op.truediv)
	core.addOp('%',"({0:s} % {1:s})","\\left({0:s} \\bmod {1:s}\\right)",False,2,op.mod)
	core.addOp('^',"({0:s} ^ {1:s})","{0:s}^{{{1:s}}}",False,1,op.pow)
	core.addOp('**',"({0:s} ^ {1:s})","{0:s}^{{{1:s}}}",False,1,op.pow)
	core.addOp('&',"({0:s} & {1:s})","\\left({0:s} \\land {1:s}\\right)",False,4,op.and_)
	core.addOp('|',"({0:s} | {1:s})","\\left({0:s} \\lor {1:s}\\right)",False,4,op.or_)
	core.addOp('</>',"({0:s} </> {1:s})","\\left({0:s} \\oplus {1:s}\\right)",False,4,op.xor)
	core.addOp('&|',"({0:s} </> {1:s})","\\left({0:s} \\oplus {1:s}\\right)",False,4,op.xor)
	core.addOp('|&',"({0:s} </> {1:s})","\\left({0:s} \\oplus {1:s}\\right)",False,4,op.xor)
	core.addOp('==',"({0:s} == {1:s})","\\left({0:s} = {1:s}\\right)",False,5,op.eq)
	core.addOp('=',"({0:s} == {1:s})","\\left({0:s} = {1:s}\\right)",False,5,op.eq)
	core.addOp('~',"({0:s} ~ {1:s})","\\left({0:s} \\approx {1:s}\\right)",False,5,sim)
	core.addOp('!~',"({0:s} !~ {1:s})","\\left({0:s} \\not\\approx {1:s}\\right)",False,5,nsim)
	core.addOp('!=',"({0:s} != {1:s})","\\left({0:s} \\neg {1:s}\\right)",False,5,op.ne)
	core.addOp('<>',"({0:s} != {1:s})","\\left({0:s} \\neg {1:s}\\right)",False,5,op.ne)
	core.addOp('><',"({0:s} != {1:s})","\\left({0:s} \\neg {1:s}\\right)",False,5,op.ne)
	core.addOp('<',"({0:s} < {1:s})","\\left({0:s} < {1:s}\\right)",False,5,op.lt)
	core.addOp('>',"({0:s} > {1:s})","\\left({0:s} > {1:s}\\right)",False,5,op.gt)
	core.addOp('<=',"({0:s} <= {1:s})","\\left({0:s} \\leq {1:s}\\right)",False,5,op.le)
	core.addOp('>=',"({0:s} >= {1:s})","\\left({0:s} \\geq {1:s}\\right)",False,5,op.ge)
	core.addOp('=<',"({0:s} <= {1:s})","\\left({0:s} \\leq {1:s}\\right)",False,5,op.le)
	core.addOp('=>',"({0:s} >= {1:s})","\\left({0:s} \\geq {1:s}\\right)",False,5,op.ge)
	core.addOp('<~',"({0:s} <~ {1:s})","\\left({0:s} \lessapprox {1:s}\\right)",False,5,lsim)
	core.addOp('>~',"({0:s} >~ {1:s})","\\left({0:s} \\gtrapprox {1:s}\\right)",False,5,gsim)
	core.addOp('~<',"({0:s} <~ {1:s})","\\left({0:s} \lessapprox {1:s}\\right)",False,5,lsim)
	core.addOp('~>',"({0:s} >~ {1:s})","\\left({0:s} \\gtrapprox {1:s}\\right)",False,5,gsim)
	core.addUnaryOp('!',"(!{0:s})","\\neg{0:s}",op.not_)
	core.addUnaryOp('-',"-{0:s}","-{0:s}",op.neg)
	core.addFn('abs',"abs({0:s})","\\left|{0:s}\\right|",1,op.abs)
	core.addFn('sum',"sum({0:s})","\\sum\\left({0:s}\\right)",'+',sumargs)
	core.addFn('prod',"prod({0:s})","\\prod\\left({0:s}\\right)",'+',product)
	if has_numpy:
		core.addFn('floor',"floor({0:s})","\\lfloor {0:s} \\rfloor",1,np.floor)
		core.addFn('ceil',"ceil({0:s})","\\lceil {0:s} \\rceil",1,np.ceil)
		core.addFn('round',"round({0:s})","\\lfloor {0:s} \\rceil",1,np.round)
		core.addFn('sin',"sin({0:s})","\\sin\\left({0:s}\\right)",1,np.sin)
		core.addFn('cos',"cos({0:s})","\\cos\\left({0:s}\\right)",1,np.cos)
		core.addFn('tan',"tan({0:s})","\\tan\\left({0:s}\\right)",1,np.tan)
		core.addFn('re',"re({0:s})","\\Re\\left({0:s}\\right)",1,np.real)
		core.addFn('im',"re({0:s})","\\Im\\left({0:s}\\right)",1,np.imag)
		core.addFn('sqrt',"sqrt({0:s})","\\sqrt{{{0:s}}}",1,np.sqrt)
		core.addConst("pi",np.pi)
		core.addConst("e",np.e)
		core.addConst("Inf",np.Inf)
		core.addConst("NaN",np.NaN)
	else:
		core.addFn('floor',"floor({0:s})","\\lfloor {0:s} \\rfloor",1,math.floor)
		core.addFn('ceil',"ceil({0:s})","\\lceil {0:s} \\rceil",1,math.ceil)
		core.addFn('round',"round({0:s})","\\lfloor {0:s} \\rceil",1,round)
		core.addFn('sin',"sin({0:s})","\\sin\\left({0:s}\\right)",1,math.sin)
		core.addFn('cos',"cos({0:s})","\\cos\\left({0:s}\\right)",1,math.cos)
		core.addFn('tan',"tan({0:s})","\\tan\\left({0:s}\\right)",1,math.tan)
		core.addFn('re',"re({0:s})","\\Re\\left({0:s}\\right)",1,complex.real)
		core.addFn('im',"re({0:s})","\\Im\\left({0:s}\\right)",1,complex.imag)
		core.addFn('sqrt',"sqrt({0:s})","\\sqrt{{{0:s}}}",1,math.sqrt)
		core.addConst("pi",math.pi)
		core.addConst("e",math.e)
		core.addConst("Inf",float("Inf"))
		core.addConst("NaN",float("NaN"))
	if has_scipy:
		core.addConst("h",scipy.constants.h)
		core.addConst("hbar",scipy.constants.hbar)
		core.addConst("m_e",scipy.constants.m_e)
		core.addConst("m_p",scipy.constants.m_p)
		core.addConst("m_n",scipy.constants.m_n)
		core.addConst("c",scipy.constants.c)
		core.addConst("N_A",scipy.constants.N_A)
		core.addConst("mu_0",scipy.constants.mu_0)
		core.addConst("eps_0",scipy.constants.epsilon_0)
		core.addConst("k",scipy.constants.k)
		core.addConst("G",scipy.constants.G)
		core.addConst("g",scipy.constants.g)
		core.addConst("q",scipy.constants.e)
		core.addConst("R",scipy.constants.R)
		core.addConst("sigma",scipy.constants.e)
		core.addConst("Rb",scipy.constants.Rydberg)