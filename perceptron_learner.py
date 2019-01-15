import numpy as np

from supervised_learner import SupervisedLearner


class PerceptronLearner(SupervisedLearner):

	def __init__(self):
		self.weights = []
		self.lr = .1
		self.bias = -1
		self.label_to_index = {}
		self.index_to_label = {}
		self.in_training = False

	def train(self, features, labels):
		print("IN TRAINING")
		self.in_training = True
		instances = self.prep_instances(features.to_numpy())
		targets = self.prep_targets(labels.to_numpy(flatten=True))
		self.weights = self.reset_weights(features.cols, targets.shape[1])
		old_loss = 0
		stagnant_epochs = 0
		while True:
			epoch_loss = 0
			for instance, target in zip(instances, targets):
				output = self.forward(instance)
				losses = target - output
				delta = self.lr * np.dot(instance.reshape(len(instance), 1), losses.reshape(1, len(losses)))
				self.weights = self.weights + delta
				epoch_loss += sum([abs(x) for x in losses])
			# print(epoch_loss)
			if epoch_loss == 0:
				break
			elif epoch_loss >= old_loss:
				stagnant_epochs += 1
				if stagnant_epochs == 5:
					break
			else:
				stagnant_epochs = 0
				old_loss = epoch_loss

	def predict(self, features, labels):
		self.in_training = False
		del labels[:]
		labels.append(self.forward(features + [1]))

	def prep_instances(self, features):
		return np.append(features, [[1]]*features.shape[0], axis=1)

	def prep_targets(self, labels):
		unique_labels = np.unique(labels)

		self.label_to_index = {k: v for k, v in zip(unique_labels, range(len(unique_labels)))}
		self.index_to_label = {k: v for k, v in zip(range(len(unique_labels)), unique_labels)}

		targets = []
		for label in labels:
			targets.append([1 if x == self.label_to_index[label] else 0 for x in range(len(unique_labels))])

		return np.asarray(targets)

	def reset_weights(self, n_features, n_classes):
		return .1 * np.random.random((n_features+1, n_classes))

	def forward(self, instance):
		net = np.dot(instance, self.weights)
		return self.activation(net)

	def activation(self, z):

		if self.in_training:
			output = np.ndarray(z.shape)
			for i, elem in enumerate(z):
				if elem > 0:
					output[i] = 1
				else:
					output[i] = 0
			return output
		else:
			return np.argmax(z)
