###############################################################################
#
#  ML example witn Iris dataset.
#  Used only numpy and python.
#
#  According to:  https://youtu.be/bXGBeRzM87g?si=KWDkyj9l-s_rPQAa
#
###############################################################################

import numpy as np

# RAND
seed = 42  # initial value should be changed
def wml_rand():
	global seed
	a = 1103515245
	c = 12345
	m = 2147483648    # 2^31, should be power of 2
	seed = (a * seed + c) % m
	return seed


def wml_rand_10():
	l = 10
	return wml_rand() % (2*l) - l


def arr_rand(arr):
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			arr[i,j] = wml_rand_10()
	return arr


def sparse_cross_entropy_batch(z, y_inds):
	# add small value to avoid log(0)
	return -np.log(np.array([z[j, y_inds[j]] for j in range(len(y_inds))]) + 1e-10)


def iris():

	import random
	import math
	import numpy as np

	INPUT_DIM = 4
	OUT_DIM = 3
	H_DIM = 10

	from sklearn import datasets
	iris = datasets.load_iris()
	dataset = [(10*iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
	if 1:
		# original
		dataset_train = dataset[ 0:40] + dataset[50: 90] + dataset[100:140]
		dataset_valid = dataset[40:50] + dataset[90:100] + dataset[140:150]
	else:
		# if we used shuffle(dataset before split to train and test
		random.shuffle(dataset)
		dataset_train = dataset[ 0:120]
		dataset_valid = dataset[120:150]
	print(f'dataset_train.len={len(dataset_train)}')
	print(f'dataset_valid.len={len(dataset_valid)}')

	if 0: # original
		W1 = np.random.randn(INPUT_DIM, H_DIM)
		b1 = np.random.randn(1, H_DIM)
		W2 = np.random.randn(H_DIM, OUT_DIM)
		b2 = np.random.randn(1, OUT_DIM)
		# change normal distribution to ravnomernoe
		W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
		b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
		W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
		b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)
	else:  # as in wml implementation
		# WA: fixed initial params to compare with C version
		W1 = np.zeros((INPUT_DIM, H_DIM))
		b1 = np.zeros((1, H_DIM))
		W2 = np.zeros((H_DIM, OUT_DIM))
		b2 = np.zeros((1, OUT_DIM))
		W1 = arr_rand(W1)
		b1 = arr_rand(b1)
		W2 = arr_rand(W2)
		b2 = arr_rand(b2)
		print("W1:\n", W1)
		print("b1:\n", b1)
		print("W2:\n", W2)
		print("b2:\n", b2)

	def relu(t):
		return np.maximum(t, 0)

	def softmax(t):
		out = np.exp(t)
		return out / np.sum(out)

	def softmax_batch(batch):
		t = np.copy(batch)
		# worm:  WA: to avoid big exp() decrease
		for i in range(len(t)):
			t[i] -= np.max(t[i])
		out = np.exp(t)
		return out / np.sum(out, axis=1, keepdims=True)

	def sparse_cross_entropy(z, y_ind):
		return -np.log(z[0, y_ind])

	def to_full_batch(y_inds, num_classes):
		y_full = np.zeros((len(y_inds), num_classes))
		for j, yj in enumerate(y_inds):
			y_full[j, yj] = 1
		return y_full

	def relu_deriv(t):
		return (t >= 0).astype(float)

	ALPHA = 0.0001
	NUM_EPOCHS = 100 # orig: 400
	BATCH_SIZE = 20

	loss_err = []

	# BATCH
	batch_cnt = 0
	for ep in range(NUM_EPOCHS):
		if 0:
			random.shuffle(dataset_train)

		for i in range(len(dataset_train) // BATCH_SIZE):

			batch_cnt += 1

			batch_x, batch_y = zip(*dataset_train[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
			x = np.concatenate(batch_x, axis=0)
			y = np.array(batch_y)

			#            ________       
			# x -> t1 -> h1 -> t2 -> z  
			#     out1  out2  out3  out4

			# forward
			t1 = x @ W1 + b1
			h1 = relu(t1)
			t2 = h1 @ W2 + b2
			z = softmax_batch(t2)  # original
			E = np.sum(sparse_cross_entropy_batch(z, y))

			# backward
			y_full = to_full_batch(y, OUT_DIM)
			dE_dt2 = z - y_full
			dE_dW2 = h1.T @ dE_dt2
			dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
			dE_dh1 = dE_dt2 @ W2.T
			dE_dt1 = dE_dh1 * relu_deriv(t1)
			dE_dW1 = x.T @ dE_dt1
			dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

			# update
			W1 = W1 - ALPHA * dE_dW1
			b1 = b1 - ALPHA * dE_db1
			W2 = W2 - ALPHA * dE_dW2
			b2 = b2 - ALPHA * dE_db2

			print(f"batch_cnt={batch_cnt}:  loss:  {E}")
			loss_err.append(E)


	def predict(x):
		t1 = x @ W1 + b1
		h1 = relu(t1)
		t2 = h1 @ W2 + b2
		z = softmax_batch(t2)
		return z

	def calc_accuracy():
		correct = 0
		for x, y in dataset_valid:
			z = predict(x)
			y_pred = np.argmax(z)
			print(f'calc_accuracy:  y={y}, y_pred={y_pred}, {"*" if y==y_pred else "-"}')
			if y_pred == y:
				correct += 1
		acc = correct / len(dataset_valid)
		return acc

	accuracy = calc_accuracy()
	print(f'Accuracy:  {accuracy}')

	# show loss error
	import matplotlib.pyplot as plt
	plt.plot(loss_err)
	plt.show()


def main():

	print('Hello')

	iris()

main()
