###############################################################################
#
#  ML example witn MNIST dataset.
#  Used PyTorch and python.
#
###############################################################################

import torch

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
print(device)

from torchvision.datasets import MNIST
import torchvision.transforms as tfs

data_tfs = tfs.Compose([
	tfs.ToTensor(),
	tfs.Normalize((0.5), (0.5))
])

# install for train and test
root = './'
train = MNIST(root, train=True,  transform=data_tfs, download=True)
test  = MNIST(root, train=False, transform=data_tfs, download=True)

print(f'Data size:')
print(f'    train     {len(train)},')
print(f'    test      {len(test)}')
print(f'Data shape:')
print(f'    features  {train[0][0].shape},')
print(f'    target    {type(test[0][1])}')

# data loaders
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)

# look at a batch
x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape, y_batch.shape)

# Learning

features = 784
classes = 10

W = torch.FloatTensor(features, classes).uniform_(-1, 1) / features**0.5
W.requires_grad_()
print(W)

# SGD learning cycle

epochs = 10
lr = 1e-2  # 1e-2 -> acc=90%, 1e-3 -> 85%, 1e-1 -> 90%
history = []

import numpy as np
from torch.nn.functional import cross_entropy

for i in range(epochs):

	for x_batch, y_batch in train_loader:

		# 1. load a batch
		x_batch = x_batch.reshape(x_batch.shape[0], -1)

		# 2. go forward
		logits = x_batch @ W
		probabilities = torch.exp(logits) / torch.exp(logits).sum(dim=1, keepdims=True)

		# 3. calc loss
		loss = -torch.log(probabilities[range(batch_size), y_batch]).mean()
		history.append(loss.item())

		# 4. go backward
		loss.backward()

		# 5. make SGD step
		grad = W.grad
		with torch.no_grad():
			W -= lr * grad
		W.grad.zero_()

	print(f'{i+1},\t loss: {history[-1]}')


# draw loss during the learning

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.plot(history)

plt.title('Loss by batch iterations')
plt.ylabel('Entropy Loss')
plt.xlabel('batches')

# show quality

from sklearn.metrics import accuracy_score

acc = 0
batches = 0

for x_batch, y_batch in test_loader:

	batches += 1
	x_batch = x_batch.view(x_batch.shape[0], -1)
	y_batch = y_batch

	preds = torch.argmax(x_batch @ W, dim=1)
	acc += (preds==y_batch).cpu().numpy().mean()

print(f'Test accuracy {acc / batches:.3}')

plt.show()
