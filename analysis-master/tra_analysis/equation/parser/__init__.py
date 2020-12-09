# Titan Robotics Team 2022: Expression submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis.Equation import parser'
# 	 adapted from https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py
# setup:

__version__ = "0.0.4-alpha"

__changelog__ = """changelog:
	0.0.4-alpha:
		- moved individual parsers to their own filespar
	0.0.3-alpha:
		- readded old regex based parser as RegexInplaceParser
	0.0.2-alpha:
		- wrote BNF using pyparsing and uses a BNF metasyntax
		- renamed this submodule parser
	0.0.1-alpha:
		- took items from equation.ipynb and ported here
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = {
	"BNF",
	"RegexInplaceParser"
}

from .BNF import BNF as BNF
from .RegexInplaceParser import RegexInplaceParser as RegexInplaceParser