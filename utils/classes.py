import numpy as np
from numpy import array
from random import shuffle


class sqds(list):

	@staticmethod
	def load(path, utf=True):
		file = open(path,'r')
		r = sqds()
		for s in file.readlines():
			if utf: s = eval(s).decode('utf-8')
			r.append(eval(s))
		file.close()
		return r

	def save(self, path, utf=True):
		file = open(path,'w+')
		for v in self:
			s = repr(v)
			if utf: s = repr(s.encode('utf-8'))
			file.write('%s\n'%s)
		file.close()

	def shuffled(self):
		r = sqds(self)
		shuffle(r)
		return r

	def split(self, m):
		if m < 1: m = int(m*len(self))
		s1 = self[:m]
		s2 = self[m:]
		return s1, s2
	
	def np2(self):
		max_len = max([len(v) for v in self])
		return np.array([np.pad(v, [0,max_len-len(v)], 'constant') for v in self])

	def numpy(self, dims=2):
		pass

class namespace():

	def __init__(self, **args):
		self.__dict__.update(args)


def class_like(parent, name=None):
	name = name if name is not None else '%s_like'%parent.__name__
	mimic = type(name, (parent,), {})
	return mimic