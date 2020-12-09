# Titan Robotics Team 2022: Expression submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis.Equation import Expression'
# 	 adapted from https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py
# setup:

__version__ = "0.0.1-alpha"

__changelog__ = """changelog:
	0.0.1-alpha:
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

import re
from .parser import BNF

class Expression():

	expression = None
	protected = list(BNF().fn.keys())

	def __init__(self, s):
		if(self.validate(s)):
			self.expression = s
		else:
			pass

	def validate(self, s):

		return true

	def substitute(self, var, value):

		pass