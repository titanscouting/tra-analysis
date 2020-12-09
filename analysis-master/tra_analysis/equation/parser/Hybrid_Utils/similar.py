_tol = 1e-5

def sim(a,b):
	if (a==b):
		return True
	elif a == 0 or b == 0:
		return False
	if (a<b):
		return (1-a/b)<=_tol
	else:
		return (1-b/a)<=_tol

def nsim(a,b):
	if (a==b):
		return False
	elif a == 0 or b == 0:
		return True
	if (a<b):
		return (1-a/b)>_tol
	else:
		return (1-b/a)>_tol
	
def gsim(a,b):
	if a >= b:
		return True
	return (1-a/b)<=_tol

def lsim(a,b):
	if a <= b:
		return True
	return (1-b/a)<=_tol
	
def set_tol(value=1e-5):
	r"""Set Error Tolerance
	
	Set the tolerance for detriming if two numbers are simliar, i.e
	:math:`\left|\frac{a}{b}\right| = 1 \pm tolerance`
	
	Parameters
	----------
	value: float
		The Value to set the tolerance to show be very small as it respresents the
		percentage of acceptable error in detriming if two values are the same.
	"""
	global _tol
	if isinstance(value,float):
		_tol = value
	else:
		raise TypeError(type(value))