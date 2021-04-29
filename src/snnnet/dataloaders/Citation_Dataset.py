# Dataset class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np

import Edge
import Vertex

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class Citation_Dataset:
	def __init__(self, class_fn, vertex_fn, edge_fn, feature_fn, meta_fn, train_fn=None, val_fn=None, test_fn=None):
		self.class_fn = class_fn
		self.vertex_fn = vertex_fn
		self.edge_fn = edge_fn
		self.feature_fn = feature_fn
		self.meta_fn = meta_fn
		self.train_fn = train_fn
		self.val_fn = val_fn
		self.test_fn = test_fn

		self.load_meta()
		self.load_edges()

		if self.train_fn != None and self.val_fn != None and self.test_fn != None:
			self.train_edges, self.train_non_edges = self.load_split(self.train_fn)
			self.val_edges, self.val_non_edges = self.load_split(self.val_fn)
			self.test_edges, self.test_non_edges = self.load_split(self.test_fn)
			assert len(self.train_edges) + len(self.val_edges) + len(self.test_edges) == len(self.edges)

		assert self.nVertices <= self.nPapers
		assert self.nNoFeatures >= 0

		self.vertices = []
		for index in range(self.nPapers):
			self.vertices.append(Vertex.Vertex(index))

		self.load_name()
		self.load_class()
		self.load_feature()
		self.validate_data()

	def load_meta(self):
		file = open(self.meta_fn, "r")
		file.readline()
		self.nVertices = int(file.readline())
		file.readline()
		self.nPapers = int(file.readline())
		file.readline()
		self.nNoFeatures = int(file.readline())
		file.readline()
		self.nEdges = int(file.readline())
		file.readline()
		self.nVocab = int(file.readline())
		file.readline()
		self.nClasses = int(file.readline())
		file.readline()
		self.classes = []
		for c in range(self.nClasses):
			self.classes.append(file.readline().strip())
		file.readline()
		self.density = float(file.readline())
		file.close()

	def load_edges(self):
		file = open(self.edge_fn, "r")
		file.readline()
		self.nCitations = int(file.readline())
		assert self.nCitations == self.nEdges
		self.edges = []
		for edge in range(self.nEdges):
			file.readline()
			index = int(file.readline())
			file.readline()
			begin = int(file.readline())
			file.readline()
			end = int(file.readline())
			self.edges.append(Edge.Edge(index, begin, end))
		file.close()

	def load_split(self, file_name):
		edges = []
		non_edges = []

		file = open(file_name, "r");
		file.readline()
		nEdges = int(file.readline())
		file.readline()
		nNonEdges = int(file.readline())
		file.readline()
		for edge in range(nEdges):
			file.readline()
			index = int(file.readline())
			file.readline()
			begin = int(file.readline())
			file.readline()
			end = int(file.readline())
			edges.append(Edge.Edge(index, begin, end))
			assert index == self.edges[index].index
			assert begin == self.edges[index].begin
			assert end == self.edges[index].end
		file.readline()
		for edge in range(nNonEdges):
			file.readline()
			begin = int(file.readline())
			file.readline()
			end = int(file.readline())
			non_edges.append(Edge.Edge(-1, begin, end))
		file.close();

		return edges, non_edges

	def edges_to_tensor(self):
		# self.edges_tensor = np.zeros((self.nEdges, 2), dtype = np.int)
		self.edges_tensor = np.zeros((self.nEdges, 2), dtype = np.float32)

		for edge in range(self.nEdges):
			self.edges_tensor[edge, 0] = self.edges[edge].begin
			self.edges_tensor[edge, 1] = self.edges[edge].end
		
		# self.edges_tensor = torch.tensor(self.edges_tensor, dtype = dtype_int, device = device)
		self.edges_tensor = torch.from_numpy(self.edges_tensor)

	def reversed_edges_to_tensor(self):
		# self.reversed_edges_tensor = np.zeros((self.nEdges, 2), dtype = np.int)
		self.reversed_edges_tensor = np.zeros((self.nEdges, 2), dtype = np.float32)

		for edge in range(self.nEdges):
			self.reversed_edges_tensor[edge, 0] = self.edges[edge].end
			self.reversed_edges_tensor[edge, 1] = self.edges[edge].begin
		
		# self.reversed_edges_tensor = torch.tensor(self.reversed_edges_tensor, dtype = dtype_int, device = device)
		self.reversed_edges_tensor = torch.from_numpy(self.reversed_edges_tensor)

	def load_name(self):
		file = open(self.vertex_fn, "r")
		file.readline()
		assert self.nPapers == int(file.readline())
		for index in range(self.nPapers):
			file.readline()
			assert index == int(file.readline())
			file.readline()
			self.vertices[index].set_name(file.readline().strip())
		file.close()

	def load_class(self):
		file = open(self.class_fn, "r")
		for example in range(self.nVertices):
			file.readline()
			index = int(file.readline())
			file.readline()
			class_ = int(file.readline())
			self.vertices[index].set_class(class_)
		file.close()

	def class_feature_vector(self):
		feature = []
		for index in range(self.nPapers):
			if self.vertices[index].has_class:
				feature.append(self.vertices[index].class_)
			else:
				feature.append(-1)
		return torch.FloatTensor(feature)

	def load_feature(self):
		file = open(self.feature_fn, "r")
		for example in range(self.nVertices):
			file.readline()
			index = int(file.readline())
			file.readline()
			nActive = int(file.readline())
			file.readline()
			vocab = [int(element) for element in file.readline().strip().split(' ')]
			assert len(vocab) == nActive
			self.vertices[index].set_vocab(vocab)

			file.readline()
			word_value = [float(element) for element in file.readline().strip().split(' ')]
			assert len(word_value) == nActive
			self.vertices[index].set_word_value(word_value)
		file.close()

	def validate_data(self):
		has_name = 0
		has_class = 0
		has_feature = 0
		for index in range(len(self.vertices)):
			if self.vertices[index].has_name == True:
				has_name += 1
			if self.vertices[index].has_class == True:
				has_class += 1
			if self.vertices[index].has_feature == True:
				has_feature += 1
		assert has_name == self.nPapers
		assert has_class == self.nVertices
		assert has_feature == self.nVertices
		# assert len(self.train) + len(self.val) + len(self.test) == self.nVertices

	def sparse_adj(self):
		x = []
		y = []
		v = []
		for edge in range(len(self.edges)):
			begin = self.edges[edge].begin
			end = self.edges[edge].end
			x.append(begin)
			y.append(end)
			v.append(1.0)
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		self.sparse_adj = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nPapers, self.nPapers]))
		return self.sparse_adj

	def sparse_incomplete_adj(self, edges):
		x = []
		y = []
		v = []
		for edge in range(len(edges)):
			begin = edges[edge].begin
			end = edges[edge].end
			x.append(begin)
			y.append(end)
			v.append(1.0)
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		return torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nPapers, self.nPapers]))

	def sparse_symm_adj(self):
		x = []
		y = []
		v = []
		for edge in range(len(self.edges)):
			begin = self.edges[edge].begin
			end = self.edges[edge].end
			x.append(begin)
			y.append(end)
			v.append(1.0)
			x.append(end)
			y.append(begin)
			v.append(1.0)
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		self.sparse_symm_adj = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nPapers, self.nPapers]))
		return self.sparse_symm_adj

	def sparse_symm_incomplete_adj(self, edges):
		x = []
		y = []
		v = []
		for edge in range(len(edges)):
			begin = edges[edge].begin
			end = edges[edge].end
			x.append(begin)
			y.append(end)
			v.append(1.0)
			x.append(end)
			y.append(begin)
			v.append(1.0)
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		return torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nPapers, self.nPapers]))

	def sparse_feature(self):
		x = []
		y = []
		v = []
		for vertex in range(self.nPapers):
			if self.vertices[vertex].has_feature == True:
				nActive = len(self.vertices[vertex].vocab)
				x += [vertex for i in range(nActive)]
				y += self.vertices[vertex].vocab
				v += self.vertices[vertex].word_value
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		self.sparse_feature = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nPapers, self.nVocab]))
		return self.sparse_feature