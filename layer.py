import numpy as np


class Layer:

	def __init__(self, in_dim, out_dim, bias, activation):
		self.has_bias = bool(bias)
		self.weights = np.random.normal(scale=.5, size=(out_dim, in_dim+bias))
		# self.weights = np.ones((out_dim, in_dim+bias))
		self.in_nodes = np.zeros(in_dim+bias)
		self.out_nodes = np.zeros(out_dim)
		self.in_grads = np.zeros(in_dim+bias)
		self.out_grads = np.zeros(out_dim)
		self.activation = activation[0]
		self.activation_prime = activation[1]
		self.loss = 0

	def forward(self, input):
		x = np.append(input, [1])
		self.in_nodes = x.copy()
		x = np.matmul(self.weights, x)
		self.out_nodes = self.activation(x)
		return self.out_nodes

	def backward(self, loss, is_output):
		self.out_grads = self.activation_prime(self.out_nodes)

		self.out_nodes = np.multiply(loss, self.out_grads)

		loss = np.array([np.dot(self.weights[:, j], self.out_nodes) for j in range(self.weights.shape[1]-self.has_bias)])

		return loss

