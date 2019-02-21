import numpy as np
from copy import deepcopy

class SGD:

	def __init__(self, lr, momentum):
		self.lr = lr
		self.momentum = momentum
		self.prev_deltas = None

	def step(self, layers):
		if not self.prev_deltas:
			self.prev_deltas = [np.zeros_like(layer.weights) for layer in layers]

		for layer, prev_deltas in zip(layers, self.prev_deltas):
			for j, in_node in enumerate(layer.in_nodes):
				for k, out_node in enumerate(layer.out_nodes):
					delta = self.lr*in_node*out_node
					layer.weights[k, j] += delta + self.momentum*prev_deltas[k, j]
