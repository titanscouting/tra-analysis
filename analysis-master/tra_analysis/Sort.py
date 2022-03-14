# Titan Robotics Team 2022: Sort submodule
# Written by Arthur Lu and James Pan
# Notes:
#    this should be imported as a python module using 'from tra_analysis import Sort'
# setup:

__version__ = "1.0.1"

__changelog__ = """changelog:
	1.0.1:
		- fixed __all__
	1.0.0:
		- ported analysis.Sort() here
		- removed classness
"""

__author__ = (
	"Arthur Lu <learthurgo@gmail.com>",
	"James Pan <zpan@imsa.edu>",
)

__all__ = [
	"quicksort",
	"mergesort",
	"introsort",
	"heapsort",
	"insertionsort",
	"timsort",
	"selectionsort",
	"shellsort",
	"bubblesort",
	"cyclesort",
	"cocktailsort",
]

import numpy as np

def quicksort(a):

	def sort(array):

		less = []
		equal = []
		greater = []

		if len(array) > 1:
			pivot = array[0]
			for x in array:
				if x < pivot:
					less.append(x)
				elif x == pivot:
					equal.append(x)
				elif x > pivot:
					greater.append(x)
			return sort(less)+equal+sort(greater) 
		else:
			return array

	return np.array(sort(a))

def mergesort(a):

	def sort(array):

		array = array

		if len(array) >1: 
			middle = len(array) // 2
			L = array[:middle]
			R = array[middle:]
	
			sort(L)
			sort(R)
	
			i = j = k = 0

			while i < len(L) and j < len(R): 
				if L[i] < R[j]: 
					array[k] = L[i] 
					i+= 1
				else: 
					array[k] = R[j] 
					j+= 1
				k+= 1

			while i < len(L): 
				array[k] = L[i] 
				i+= 1
				k+= 1
			
			while j < len(R): 
				array[k] = R[j] 
				j+= 1
				k+= 1

			return array

	return sort(a)

def introsort(a):

	def sort(array, start, end, maxdepth):

		array = array

		if end - start <= 1:
			return
		elif maxdepth == 0:
			heapsort(array, start, end)
		else:
			p = partition(array, start, end)
			sort(array, start, p + 1, maxdepth - 1)
			sort(array, p + 1, end, maxdepth - 1)

		return array

	def partition(array, start, end):
		pivot = array[start]
		i = start - 1
		j = end
	
		while True:
			i = i + 1
			while array[i] < pivot:
				i = i + 1
			j = j - 1
			while array[j] > pivot:
				j = j - 1
	
			if i >= j:
				return j
	
			swap(array, i, j)

	def swap(array, i, j):
		array[i], array[j] = array[j], array[i]

	def heapsort(array, start, end):
		build_max_heap(array, start, end)
		for i in range(end - 1, start, -1):
			swap(array, start, i)
			max_heapify(array, index=0, start=start, end=i)

	def build_max_heap(array, start, end):
		def parent(i):
			return (i - 1)//2
		length = end - start
		index = parent(length - 1)
		while index >= 0:
			max_heapify(array, index, start, end)
			index = index - 1

	def max_heapify(array, index, start, end):
		def left(i):
			return 2*i + 1
		def right(i):
			return 2*i + 2
	
		size = end - start
		l = left(index)
		r = right(index)
		if (l < size and array[start + l] > array[start + index]):
			largest = l
		else:
			largest = index
		if (r < size and array[start + r] > array[start + largest]):
			largest = r
		if largest != index:
			swap(array, start + largest, start + index)
			max_heapify(array, largest, start, end)

	maxdepth = (len(a).bit_length() - 1)*2

	return sort(a, 0, len(a), maxdepth)

def heapsort(a):

	def sort(array):
		
		array = array

		n = len(array) 

		for i in range(n//2 - 1, -1, -1): 
			heapify(array, n, i) 

		for i in range(n-1, 0, -1): 
			array[i], array[0] = array[0], array[i]
			heapify(array, i, 0) 

		return array

	def heapify(array, n, i):

		array = array

		largest = i
		l = 2 * i + 1
		r = 2 * i + 2

		if l < n and array[i] < array[l]: 
			largest = l 

		if r < n and array[largest] < array[r]: 
			largest = r 

		if largest != i: 
			array[i],array[largest] = array[largest],array[i]
			heapify(array, n, largest)
		
		return array

	return sort(a)

def insertionsort(a):

	def sort(array):

		array = array

		for i in range(1, len(array)): 
	
			key = array[i] 

			j = i-1
			while j >=0 and key < array[j] : 
					array[j+1] = array[j] 
					j -= 1
			array[j+1] = key 

		return array

	return sort(a)

def timsort(a, block = 32):

	BLOCK = block

	def sort(array, n):

		array = array

		for i in range(0, n, BLOCK):  
			insertionsort(array, i, min((i+31), (n-1)))

		size = BLOCK 
		while size < n:

			for left in range(0, n, 2*size):  
	
				mid = left + size - 1 
				right = min((left + 2*size - 1), (n-1))  
				merge(array, left, mid, right)  
		
			size = 2*size

		return array

	def insertionsort(array, left, right):

		array = array 

		for i in range(left + 1, right+1):  
		
			temp = array[i]  
			j = i - 1 
			while j >= left and array[j] > temp :  
			
				array[j+1] = array[j]  
				j -= 1
			
			array[j+1] = temp

		return array
		

	def merge(array, l, m, r): 
	
		len1, len2 =  m - l + 1, r - m  
		left, right = [], []  
		for i in range(0, len1):  
			left.append(array[l + i])  
		for i in range(0, len2):  
			right.append(array[m + 1 + i])  
		
		i, j, k = 0, 0, l 

		while i < len1 and j < len2:  
		
			if left[i] <= right[j]:  
				array[k] = left[i]  
				i += 1 
			
			else: 
				array[k] = right[j]  
				j += 1 
			
			k += 1

		while i < len1:  
		
			array[k] = left[i]  
			k += 1 
			i += 1

		while j < len2:  
			array[k] = right[j]  
			k += 1
			j += 1

	return sort(a, len(a))

def selectionsort(a):
	array = a
	for i in range(len(array)): 
		min_idx = i
		for j in range(i+1, len(array)): 
			if array[min_idx] > array[j]: 
				min_idx = j         
		array[i], array[min_idx] = array[min_idx], array[i]
	return array

def shellsort(a):
	array = a
	n = len(array)
	gap = n//2

	while gap > 0: 

		for i in range(gap,n): 

			temp = array[i] 
			j = i 
			while  j >= gap and array[j-gap] >temp: 
				array[j] = array[j-gap] 
				j -= gap 
			array[j] = temp 
		gap //= 2

	return array

def bubblesort(a):

	def sort(array):
		for i, num in enumerate(array):
			try:
				if array[i+1] < num:
					array[i] = array[i+1]
					array[i+1] = num
					sort(array)
			except IndexError:
				pass
		return array

	return sort(a)

def cyclesort(a):

	def sort(array):

		array = array
		writes = 0

		for cycleStart in range(0, len(array) - 1): 
			item = array[cycleStart] 

			pos = cycleStart 
			for i in range(cycleStart + 1, len(array)): 
				if array[i] < item: 
					pos += 1

			if pos == cycleStart: 
				continue

			while item == array[pos]: 
				pos += 1
				array[pos], item = item, array[pos] 
				writes += 1

			while pos != cycleStart: 

				pos = cycleStart 
				for i in range(cycleStart + 1, len(array)): 
					if array[i] < item: 
						pos += 1

				while item == array[pos]: 
					pos += 1
				array[pos], item = item, array[pos] 
				writes += 1
			
		return array
	
	return sort(a)

def cocktailsort(a):

	def sort(array):

		array = array

		n = len(array) 
		swapped = True
		start = 0
		end = n-1
		while (swapped == True): 
			swapped = False
			for i in range (start, end): 
				if (array[i] > array[i + 1]) : 
					array[i], array[i + 1]= array[i + 1], array[i] 
					swapped = True
			if (swapped == False): 
				break
			swapped = False
			end = end-1
			for i in range(end-1, start-1, -1): 
				if (array[i] > array[i + 1]): 
					array[i], array[i + 1] = array[i + 1], array[i] 
					swapped = True
			start = start + 1

		return array

	return sort(a)