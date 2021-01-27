import math
import sys
import re

if sys.version_info >= (3,):
	xrange = range
	basestring = str

class ExpressionObject(object):
	def __init__(self,*args,**kwargs):
		super(ExpressionObject,self).__init__(*args,**kwargs)

	def toStr(self,args,expression):
		return ""

	def toRepr(self,args,expression):
		return ""

	def __call__(self,args,expression):
		pass

class ExpressionValue(ExpressionObject):
	def __init__(self,value,*args,**kwargs):
		super(ExpressionValue,self).__init__(*args,**kwargs)
		self.value = value

	def toStr(self,args,expression):
		if (isinstance(self.value,complex)):
			V = [self.value.real,self.value.imag]
			E = [0,0]
			B = [0,0]
			out = ["",""]
			for i in xrange(2):
				if V[i] == 0:
					E[i] = 0
					B[i] = 0
				else:
					E[i] = int(math.floor(math.log10(abs(V[i]))))
					B[i] = V[i]*10**-E[i]
					if E[i] in [0,1,2,3] and str(V[i])[-2:] == ".0":
						B[i] = int(V[i])
						E[i] = 0
					if E[i] in [-1,-2] and len(str(V[i])) <= 7:
						B[i] = V[i]
						E[i] = 0
				if i == 1:
					fmt = "{{0:+{0:s}}}"
				else:
					fmt = "{{0:-{0:s}}}"
				if type(B[i]) == int:
					out[i] += fmt.format('d').format(B[i])
				else:
					out[i] += fmt.format('.5f').format(B[i]).rstrip("0.")
				if i == 1:
					out[i] += "\\imath"
				if E[i] != 0:
					out[i] += "\\times10^{{{0:d}}}".format(E[i])
			return "\\left(" + ''.join(out) + "\\right)"
		elif (isinstance(self.value,float)):
			V = self.value
			E = 0
			B = 0
			out = ""
			if V == 0:
				E = 0
				B = 0
			else:
				E = int(math.floor(math.log10(abs(V))))
				B = V*10**-E
				if E in [0,1,2,3] and str(V)[-2:] == ".0":
					B = int(V)
					E = 0
				if E in [-1,-2] and len(str(V)) <= 7:
					B = V
					E = 0
			if type(B) == int:
				out += "{0:-d}".format(B)
			else:
				out += "{0:-.5f}".format(B).rstrip("0.")
			if E != 0:
				out += "\\times10^{{{0:d}}}".format(E)
				return "\\left(" + out + "\\right)"
			else:
				return out
		else:
			return str(self.value)

	def toRepr(self,args,expression):
		return str(self.value)

	def __call__(self,args,expression):
		return self.value

	def __repr__(self):
			return "<{0:s}.{1:s}({2:s}) object at {3:0=#10x}>".format(type(self).__module__,type(self).__name__,str(self.value),id(self))

class ExpressionFunction(ExpressionObject):
	def __init__(self,function,nargs,form,display,id,isfunc,*args,**kwargs):
		super(ExpressionFunction,self).__init__(*args,**kwargs)
		self.function = function
		self.nargs = nargs
		self.form = form
		self.display = display
		self.id = id
		self.isfunc = isfunc

	def toStr(self,args,expression):
		params = []
		for i in xrange(self.nargs):
			params.append(args.pop())
		if self.isfunc:
			return str(self.display.format(','.join(params[::-1])))
		else:
			return str(self.display.format(*params[::-1]))

	def toRepr(self,args,expression):
		params = []
		for i in xrange(self.nargs):
			params.append(args.pop())
		if self.isfunc:
			return str(self.form.format(','.join(params[::-1])))
		else:
			return str(self.form.format(*params[::-1]))

	def __call__(self,args,expression):
		params = []
		for i in xrange(self.nargs):
			params.append(args.pop())
		return self.function(*params[::-1])

	def __repr__(self):
		return "<{0:s}.{1:s}({2:s},{3:d}) object at {4:0=#10x}>".format(type(self).__module__,type(self).__name__,str(self.id),self.nargs,id(self))

class ExpressionVariable(ExpressionObject):
	def __init__(self,name,*args,**kwargs):
		super(ExpressionVariable,self).__init__(*args,**kwargs)
		self.name = name

	def toStr(self,args,expression):
		return str(self.name)

	def toRepr(self,args,expression):
		return str(self.name)

	def __call__(self,args,expression):
		if self.name in expression.variables:
			return expression.variables[self.name]
		else:
			return 0 # Default variables to return 0

	def __repr__(self):
		return "<{0:s}.{1:s}({2:s}) object at {3:0=#10x}>".format(type(self).__module__,type(self).__name__,str(self.name),id(self))

class Core():

	constants = {}
	unary_ops = {}
	ops = {}
	functions = {}
	smatch = re.compile(r"\s*,")
	vmatch = re.compile(r"\s*"
						"(?:"
							"(?P<oct>"
								"(?P<octsign>[+-]?)"
								r"\s*0o"
								"(?P<octvalue>[0-7]+)"
							")|(?P<hex>"
								"(?P<hexsign>[+-]?)"
								r"\s*0x"
								"(?P<hexvalue>[0-9a-fA-F]+)"
							")|(?P<bin>"
								"(?P<binsign>[+-]?)"
								r"\s*0b"
								"(?P<binvalue>[01]+)"
							")|(?P<dec>"
								"(?P<rsign>[+-]?)"
								r"\s*"
								r"(?P<rvalue>(?:\d+\.\d+|\d+\.|\.\d+|\d+))"
								"(?:"
									"[Ee]"
									r"(?P<rexpoent>[+-]?\d+)"
								")?"
								"(?:"
									r"\s*"
									r"(?P<sep>(?(rvalue)\+|))?"
									r"\s*"
									"(?P<isign>(?(rvalue)(?(sep)[+-]?|[+-])|[+-]?)?)"
									r"\s*"
									r"(?P<ivalue>(?:\d+\.\d+|\d+\.|\.\d+|\d+))"
									"(?:"
										"[Ee]"
										r"(?P<iexpoent>[+-]?\d+)"
									")?"
									"[ij]"
								")?"
							")"
						")")
	nmatch = re.compile(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)")
	gsmatch = re.compile(r'\s*(\()')
	gematch = re.compile(r'\s*(\))')

	def recalculateFMatch(self):
		
		fks = sorted(self.functions.keys(), key=len, reverse=True)
		oks = sorted(self.ops.keys(), key=len, reverse=True)
		uks = sorted(self.unary_ops.keys(), key=len, reverse=True)
		self.fmatch = re.compile(r'\s*(' + '|'.join(map(re.escape,fks)) + ')')
		self.omatch = re.compile(r'\s*(' + '|'.join(map(re.escape,oks)) + ')')
		self.umatch = re.compile(r'\s*(' + '|'.join(map(re.escape,uks)) + ')')

	def addFn(self,id,str,latex,args,func):
		self.functions[id] = {
			'str': str,
			'latex': latex,
			'args': args,
			'func': func}

	def addOp(self,id,str,latex,single,prec,func):
		if single:
			raise RuntimeError("Single Ops Not Yet Supported")
		self.ops[id] = {
			'str': str,
			'latex': latex,
			'args': 2,
			'prec': prec,
			'func': func}

	def addUnaryOp(self,id,str,latex,func):
		self.unary_ops[id] = {
			'str': str,
			'latex': latex,
			'args': 1,
			'prec': 0,
			'func': func}
	
	def addConst(self,name,value):
		self.constants[name] = value