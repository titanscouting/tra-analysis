# Titan Robotics Team 2022: Array submodule
# Written by Arthur Lu
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Array'
# setup:

__version__ = "1.0.0"

__changelog__ = """changelog:
	1.0.0:
		- ported analysis.Array() here
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
)

import numpy as np

class Array(): # tests on nd arrays independent of basic_stats

	def __init__(self, narray):

		self.array = np.array(narray)

	def __str__(self):

		return str(self.array)
	
	def elementwise_mean(self, axis = 0): # expects arrays that are size normalized

		return np.mean(self.array, axis = axis)

	def elementwise_median(self, axis = 0):

		return np.median(self.array, axis = axis)

	def elementwise_stdev(self, axis = 0):

		return np.std(self.array, axis = axis)

	def elementwise_variance(self, axis = 0):

		return np.var(self.array, axis = axis)

	def elementwise_npmin(self, axis = 0):
		return np.amin(self.array, axis = axis)


	def elementwise_npmax(self, axis = 0):
		return np.amax(self.array, axis = axis)

	def elementwise_stats(self, axis = 0):

		_mean = self.elementwise_mean(axis = axis)
		_median = self.elementwise_median(axis = axis)
		_stdev = self.elementwise_stdev(axis = axis)
		_variance = self.elementwise_variance(axis = axis)
		_min = self.elementwise_npmin(axis = axis)
		_max = self.elementwise_npmax(axis = axis)

		return _mean, _median, _stdev, _variance, _min, _max

	def __getitem__(self, key):

		return self.array[key]

	def __setitem__(self, key, value):

		self.array[key] = value

	def normalize(self, array):

		a = np.atleast_1d(np.linalg.norm(array))
		a[a==0] = 1
		return array / np.expand_dims(a, -1)

	def __add__(self, other):

		return self.array + other.array

	def __sub__(self, other):

		return self.array - other.array

	def __neg__(self):
		
		return -self.array

	def __abs__(self):

		return abs(self.array)

	def __invert__(self):

		return 1/self.array

	def __mul__(self, other):

		return self.array.dot(other.array)

	def __rmul__(self, other):

		return self.array.dot(other.array)

	def cross(self, other):

		return np.cross(self.array, other.array)

	def sort(self, array): # depreciated
		warnings.warn("Array.sort has been depreciated in favor of Sort")
		array_length = len(array)
		if array_length <= 1:
			return array
		middle_index = int(array_length / 2)
		left = array[0:middle_index]
		right = array[middle_index:]
		left = self.sort(left)
		right = self.sort(right)
		return self.__merge(left, right)


	def __merge(self, left, right):
		sorted_list = []
		left = left[:]
		right = right[:]
		while len(left) > 0 or len(right) > 0:
			if len(left) > 0 and len(right) > 0:
				if left[0] <= right[0]:
					sorted_list.append(left.pop(0))
				else:
					sorted_list.append(right.pop(0))
			elif len(left) > 0:
				sorted_list.append(left.pop(0))
			elif len(right) > 0:
				sorted_list.append(right.pop(0))
		return sorted_list

	def search(self, arr, x):
		return self.__search(arr, 0, len(arr) - 1, x)

	def __search(self, arr, low, high, x): 
		if high >= low: 
			mid = (high + low) // 2
			if arr[mid] == x: 
				return mid 
			elif arr[mid] > x: 
				return binary_search(arr, low, mid - 1, x) 
			else: 
				return binary_search(arr, mid + 1, high, x) 
		else:
			return -1
