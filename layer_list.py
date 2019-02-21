import numpy as np
from layer import Layer


class LayerList:

	def __init__(self):
		self.layers = []
		self.weights = [layer.weights for layer in self.layers] if len(self.layers) else []

	def __len__(self):
		return len(self.layers)

	def __getitem__(self, item):
		return self.layers[item]

	def add_layer(self, in_dim, out_dim, activation, bias=True):
		self.layers.append(
			Layer(in_dim, out_dim, bias, activation))

	def backward(self, loss):
		for i, layer in enumerate(self.layers[::-1]):
			if i == 0:
				loss = layer.backward(loss, is_output=True)
			else:
				loss = layer.backward(loss, is_output=False)

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x
