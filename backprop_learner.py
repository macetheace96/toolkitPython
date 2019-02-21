from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from supervised_learner import SupervisedLearner
from layer_list import LayerList
from optimizer import SGD


def sigmoid(x):
	return 1/(1+np.exp(-x))


def anti_sigmoid(x):
	return x*(1-x)


def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))


def anti_softmax(x):
	return 1


def cross_entropy(z, t):
	return -np.sum(t*np.log(z))


class BackPropLearner(SupervisedLearner):

	def __init__(self):
		self.lr = .01
		self.momentum = .9

		self.n_layers = 1
		self.hidden_dim = 8

		self.val_split = .2
		self.encoder = None

		self.threshold = 50
		self.max_epochs = 3000
		self.allowance = .0

		self.hidden_activation = (sigmoid, anti_sigmoid)
		# self.output_activation = (softmax, anti_sigmoid)
		self.output_activation = (sigmoid, anti_sigmoid)
		# self.loss_function = cross_entropy
		self.loss_function = lambda z, t: t-z
		self.opt = None
		self.layers = None

	def train(self, features, labels):
		in_dim = features.cols
		out_dim = labels.value_count(0)

		full_x, full_y = self.prep_data(features, labels)
		train_x, train_y, val_x, val_y = self.split_data(full_x, full_y, self.val_split)

		self.layers = self.init_layers(in_dim, out_dim)
		best_weights = deepcopy(self.layers)

		self.opt = SGD(self.lr, self.momentum)

		train_losses = []
		train_accuracies = []
		val_losses = []
		val_accuracies = []
		lowest_loss = np.inf
		highest_accuracy = 0
		stagnant_rounds = 0
		n_epochs = 0
		try:
			while True:

				train_x, train_y = self.shuffle(train_x, train_y)

				self.run_epoch(train_x, train_y)

				train_loss, train_accuracy = self.score(train_x, train_y)
				val_loss, val_accuracy = self.score(val_x, val_y)
				train_losses.append(train_loss)
				train_accuracies.append(train_accuracy)
				val_losses.append(val_loss)
				val_accuracies.append(val_accuracy)
				print(f"EPOCH {n_epochs}")
				print(f"Train:\t{train_losses[-1]}\t{train_accuracies[-1]}")
				print(f"Val:  \t{val_losses[-1]}\t{val_accuracies[-1]}")
				print()

				if val_losses[-1] < lowest_loss + self.allowance*lowest_loss:
					lowest_loss = val_losses[-1]
					best_weights = deepcopy(self.layers)

				elif stagnant_rounds < self.threshold:
					stagnant_rounds += 1
				else:
					break

				n_epochs += 1
		except KeyboardInterrupt:
			pass
		finally:
			self.layers = best_weights

			fig, ax1 = plt.subplots()

			ax1.set_title("Iris Validation Set Loss vs Accuracy")

			color = 'tab:red'
			ax1.set_xlabel('Epochs')

			ax1.set_ylabel('Loss (MSE)', color=color)
			ax1.plot(val_losses, color=color)
			ax1.tick_params(axis='y', labelcolor=color)

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

			color = 'tab:blue'
			ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
			ax2.plot(val_accuracies, color=color)
			ax2.tick_params(axis='y', labelcolor=color)

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			# plt.savefig("/Users/masonfp/Desktop/cs/CS478-Machine-Learning-Projects/plots/backprop/iris.png")
			plt.show()

	def shuffle(self, a, b):
		temp = list(zip(a, b))
		np.random.shuffle(temp)
		new_a, new_b = zip(*temp)
		return np.array(new_a), np.array(new_b)

	def prep_data(self, features, labels):
		instances = features.to_numpy()

		if not self.encoder:
			self.encoder = OneHotEncoder(sparse=False, categories='auto')
		targets = self.encoder.fit_transform(labels.data)

		return instances, targets

	def split_data(self, x, y, split):
		n_samples = int(len(x) * split)
		rand_indices = np.random.permutation(range(len(x)))

		old_x = x[rand_indices[n_samples:]]
		old_y = y[rand_indices[n_samples:]]
		new_x = x[rand_indices[:n_samples]]
		new_y = y[rand_indices[:n_samples]]

		return old_x, old_y, new_x, new_y

	def score(self, X, Y):
		losses = []
		accuracy_count = 0
		for x, y in zip(X, Y):
			logits = self.layers.forward(x)
			loss = self.loss_function(logits, y)
			accuracy_count += np.argmax(y) == np.argmax(logits)
			losses.append(loss**2)

		return np.mean(losses), accuracy_count/len(Y)

	def run_epoch(self, X, Y):
		losses = []
		for x, y in zip(X, Y):
			logits = self.layers.forward(x)
			loss = self.loss_function(logits, y)
			losses.append(loss)
			self.layers.backward(loss)
			self.opt.step(self.layers)
		return np.mean(losses)

	def init_layers(self, in_dim, out_dim):
		layers = LayerList()

		for x in range(self.n_layers):
			if x == 0:
				layers.add_layer(in_dim, self.hidden_dim, self.hidden_activation)
			else:
				layers.add_layer(self.hidden_dim, self.hidden_dim, self.hidden_activation)
		if len(layers):
			layers.add_layer(self.hidden_dim, out_dim, self.output_activation)
		else:
			layers.add_layer(in_dim, out_dim, self.output_activation)

		return layers

	def predict(self, features, labels):
		self.in_training = False
		del labels[:]
		pred = self.layers.forward(np.array(features))
		pred = [[1 if x == max(pred) else 0 for x in pred]]
		pred = self.encoder.inverse_transform(pred)
		labels.append(pred[0][0])

