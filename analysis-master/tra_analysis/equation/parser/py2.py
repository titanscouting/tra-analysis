# Titan Robotics Team 2022: py2 module
# Written by Arthur Lu
# Notes:
#    this module should only be used internally, contains old python 2.X functions that have been removed.
# setup:

from __future__ import division

__version__ = "1.0.0"

__changelog__ = """changelog:
	1.0.0:
	- added cmp function
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

def cmp(a, b):
    return (a > b) - (a < b) 