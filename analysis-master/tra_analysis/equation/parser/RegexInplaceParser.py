import re
from decimal import Decimal
from functools import reduce

class RegexInplaceParser(object):

	def __init__(self, string):

		self.string = string

	def add(self, string):
		while(len(re.findall("[+]{1}[-]?", string)) != 0):
			string = re.sub("[-]?\d+[.]?\d*[+]{1}[-]?\d+[.]?\d*", str("%f" % reduce((lambda x, y: x + y), [Decimal(i) for i in re.split("[+]{1}", re.search("[-]?\d+[.]?\d*[+]{1}[-]?\d+[.]?\d*", string).group())])), string, 1)
		return string

	def sub(self, string):
		while(len(re.findall("\d+[.]?\d*[-]{1,2}\d+[.]?\d*", string)) != 0):
			g = re.search("\d+[.]?\d*[-]{1,2}\d+[.]?\d*", string).group()
			if(re.search("[-]{1,2}", g).group() == "-"):
				r = re.sub("[-]{1}", "+-", g, 1)
				string = re.sub(g, r, string, 1)
			elif(re.search("[-]{1,2}", g).group() == "--"):
				r = re.sub("[-]{2}", "+", g, 1)
				string = re.sub(g, r, string, 1)
			else:
				pass
		return string

	def mul(self, string):
		while(len(re.findall("[*]{1}[-]?", string)) != 0):
			string = re.sub("[-]?\d+[.]?\d*[*]{1}[-]?\d+[.]?\d*", str("%f" % reduce((lambda x, y: x * y), [Decimal(i) for i in re.split("[*]{1}", re.search("[-]?\d+[.]?\d*[*]{1}[-]?\d+[.]?\d*", string).group())])), string, 1)
		return string

	def div(self, string):
		while(len(re.findall("[/]{1}[-]?", string)) != 0):
			string = re.sub("[-]?\d+[.]?\d*[/]{1}[-]?\d+[.]?\d*", str("%f" % reduce((lambda x, y: x / y), [Decimal(i) for i in re.split("[/]{1}", re.search("[-]?\d+[.]?\d*[/]{1}[-]?\d+[.]?\d*", string).group())])), string, 1)
		return string

	def exp(self, string):
		while(len(re.findall("[\^]{1}[-]?", string)) != 0):
			string = re.sub("[-]?\d+[.]?\d*[\^]{1}[-]?\d+[.]?\d*", str("%f" % reduce((lambda x, y: x ** y), [Decimal(i) for i in re.split("[\^]{1}", re.search("[-]?\d+[.]?\d*[\^]{1}[-]?\d+[.]?\d*", string).group())])), string, 1)
		return string

	def evaluate(self):
		string = self.string
		string = self.exp(string)
		string = self.div(string)
		string = self.mul(string)
		string = self.sub(string)
		string = self.add(string)
		return string