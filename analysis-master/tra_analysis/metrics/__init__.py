# Titan Robotics Team 2022: Metrics submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import metrics'
# setup:

__version__ = "1.0.0"

__changelog__ = """changelog:
	1.0.0:
		- implemented elo, glicko2, trueskill
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

__all__ = {
	"Expression"
}

from . import elo
from . import glicko2
from . import trueskill