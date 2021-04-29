# Titan Robotics Team 2022: Expression submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Equation'
# setup:

__version__ = "0.0.1-alpha"

__changelog__ = """changelog:
	0.0.1-alpha:
		- made first prototype of Expression
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = {
    "Expression"
}

from .Expression import Expression