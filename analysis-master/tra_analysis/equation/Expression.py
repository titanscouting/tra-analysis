# Titan Robotics Team 2022: Expression submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis.Equation import Expression'
# 	 TODO:
#	 	 - add option to pick parser backend
#		 - fix unit tests
# setup:

__version__ = "0.0.1-alpha"

__changelog__ = """changelog:
	0.0.1-alpha:
		- used the HybridExpressionParser as backend for Expression
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = {
	"Expression"
}

import re
from .parser import BNF, RegexInplaceParser, HybridExpressionParser, Core, equation_base

class Expression(HybridExpressionParser):

	expression = None
	core = None

	def __init__(self,expression,argorder=[],*args,**kwargs):
		self.core = Core()
		equation_base.equation_extend(self.core)
		self.core.recalculateFMatch()
		super().__init__(self.core, expression, argorder=[],*args,**kwargs)