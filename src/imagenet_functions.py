import os
import re
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from densenet import DenseNet121
from tensorflow.keras.regularizers import l2
from tensorflow.python.client import device_lib
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import top_k_accuracy_score, accuracy_score

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def load_imagenet(batches = 10):
	if batches > 10 or batches < 1:
		batches = 10

	path_train = '../datasets/imagenet/32/train_data_batch_nhwc_'
	train_x = None
	train_y = []

	# Load train batches
	for i in range(1, 1 + batches):
		print(f"Loading batch {i}")
		d = unpickle(f'{path_train}{i}')
		data = d['data']
		labels = d['labels']
		mean = d['mean']
		stdev = d['stdev']

		if train_x is None:
			train_x = (data - mean) / stdev
		else:
			train_x = np.vstack(( train_x, (data - mean) / stdev ))

		train_y = train_y + labels

	train_y = np.array(train_y) - 1

	# Load test
	path_test = '../datasets/imagenet/32/val_data_nhwc'
	d = unpickle(path_test)
	data = d['data']
	labels = d['labels']
	test_x = (data - mean) / stdev
	test_y = np.array(labels) - 1

	return train_x, train_y, test_x, test_y

def load_imagenet224(batches = -1, val_split=0.05, seed=1):
	def _parse_image(example):
		parsed = tf.io.parse_single_example(example, features)
		image = (tf.cast(tf.io.decode_jpeg(parsed['image']), dtype=tf.float32) - train_mean) / 128.0
		label = parsed['label']
		return image, label

	# Load pickle with train mean
	with open('../datasets/imagenet/224/train_mean.pkl', 'rb') as f:
		train_mean = pickle.load(f)
	
	features = {
		'image' : tf.io.FixedLenFeature([], tf.string),
		'label' : tf.io.FixedLenFeature([], tf.int64)
	}

	path_train = Path('../datasets/imagenet/224/train')

	filenames = []
	count = 0
	for fname in os.listdir(path_train):
		if fname.endswith('.tfrecords') and (count < batches or batches < 0):
			filenames.append(str(path_train / fname))
			count += 1

	print("Loading train...")
	full_train_dataset = tf.data.TFRecordDataset(filenames, buffer_size=100, num_parallel_reads=5).shuffle(4096, seed=seed)
	dataset_size = 0
	for example in full_train_dataset:
		dataset_size += 1

	train_size = int(dataset_size * (1.0 - val_split))
	val_size = dataset_size - train_size
	print("Splitting train and validation...")
	train_dataset = full_train_dataset.take(train_size).shuffle(4096, seed=seed).map(_parse_image, num_parallel_calls=6)
	val_dataset = full_train_dataset.skip(train_size).map(_parse_image, num_parallel_calls=6)

	path_test = Path('../datasets/imagenet/224/val')

	filenames = []
	for fname in os.listdir(path_test):
		if fname.endswith('.tfrecords'):
			filenames.append(str(path_test / fname))

	print("Loading test...")
	test_dataset = tf.data.TFRecordDataset(filenames, buffer_size=100, num_parallel_reads=5).map(_parse_image, num_parallel_calls=6)
	test_size = 0
	for example in test_dataset:
		test_size += 1

	return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size


def compute_metrics(y_true, y_pred, num_classes):
	# Calculate metric
	acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
	top2 = top_k_accuracy_score(y_true, y_pred, k=2, labels=range(num_classes))
	top3 = top_k_accuracy_score(y_true, y_pred, k=3, labels=range(num_classes))

	metrics = {
		'CCR': acc,
		'Top-2': top2,
		'Top-3': top3
	}

	return metrics

def print_metrics(metrics):
	print('CCR: {:.4f}'.format(metrics['CCR']))
	print('Top-2: {:.4f}'.format(metrics['Top-2']))
	print('Top-3: {:.4f}'.format(metrics['Top-3']))

class spp(keras.layers.Layer):
	"""
	Parametric softplus activation layer.
	"""

	def __init__(self, alpha, **kwargs):
		super(spp, self).__init__(**kwargs)
		self.__name__ = 'spp'
		self.alpha = alpha

	def build(self, input_shape):
		super(spp, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return tf.nn.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape

class sppt(keras.layers.Layer):
	"""
	Trainable Parametric softplus activation layer.
	"""

	def __init__(self, **kwargs):
		super(sppt, self).__init__(**kwargs)
		self.__name__ = 'sppt'

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=tf.float32,
									 initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
									 trainable=True)

		super(sppt, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return tf.nn.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape

class eluplus(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(eluplus, self).__init__(**kwargs)

	def build(self, input_shape):
		self.lmbd = self.add_weight(name='lambda', shape=(int(input_shape[-1]),), dtype=tf.float32,
									 initializer=keras.initializers.Constant(value=0.5))

		self.lmbd = tf.clip_by_value(self.lmbd, 0.0, 1.0)

		super(eluplus, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return self.lmbd * tf.nn.elu(inputs) + (1 - self.lmbd) * tf.nn.softplus(inputs)

class elusppt(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(elusppt, self).__init__(**kwargs)

	def build(self, input_shape):
		self.lmbd = self.add_weight(name='lambda', shape=(int(input_shape[-1]),), dtype=tf.float32,
									 initializer=keras.initializers.Constant(value=0.5))

		self.lmbd = tf.clip_by_value(self.lmbd, 0.0, 1.0)

		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=tf.float32,
									 initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
									 trainable=True)

		super(elusppt, self).build(input_shape)

	def call(self, inputs, **kwargs):
		output = self.lmbd * tf.nn.elu(inputs) + (1 - self.lmbd) * (tf.nn.softplus(inputs) - self.alpha)

		return output

def get_activation(name):
	try:
		# Try to get activation from keras standard activation
		activation = lambda: keras.layers.Activation(name)
		activation()
	except:
		# Error means that activation does not exist. Find a class with its name in this module.
		if name in globals():
			activation = lambda: globals()[name]()
		else:
			print(f'WARNING: Activation function {name} could not be found. Falling back to default ReLU')
			activation = lambda: keras.layers.Activation('relu')
	
	return activation

def create_vgg16(input_shape, num_classes, activation_name = 'relu'):
	# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

	# Weight decay for L2 regularizer
	weight_decay = 0.0005

	activation = get_activation(activation_name)

	l_in = keras.layers.Input(shape=input_shape)
	
	# Block 1
	layer = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(l_in)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.3, seed=1)(layer)

	layer = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.MaxPooling2D(pool_size=(2,2))(layer)

	# Block 2
	layer = keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.MaxPooling2D(pool_size=(2,2))(layer)

	# Block 3
	layer = keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.MaxPooling2D(pool_size=(2,2))(layer)

	# Block 4
	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.MaxPooling2D(pool_size=(2,2))(layer)

	# Block 5
	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.Dropout(0.4, seed=1)(layer)

	layer = keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(weight_decay))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)
	layer = keras.layers.MaxPooling2D(pool_size=(2,2))(layer)
	layer = keras.layers.Dropout(0.5, seed=1)(layer)

	layer = keras.layers.GlobalAveragePooling2D()(layer)

	# Dense 1
	layer = keras.layers.Dense(512, kernel_regularizer=l2(weight_decay), kernel_initializer=glorot_uniform(seed=1))(layer)
	layer = activation()(layer)
	layer = keras.layers.BatchNormalization()(layer)

	layer = keras.layers.Dropout(0.5, seed=1)(layer)
	out = keras.layers.Dense(num_classes, dtype=tf.float32, kernel_initializer=glorot_uniform(seed=1))(layer)
	out = keras.layers.Activation('softmax', dtype=tf.float32)(out)

	model = keras.models.Model(l_in, out, name='VGG16')

	return model

def create_densenet121(input_shape, num_classes, activation_name = 'relu'):
	activation = get_activation(activation_name)

	l_in = keras.layers.Input(input_shape)
	l_in, out = DenseNet121(input_tensor=l_in, pooling='avg', activation=activation)
	out = keras.layers.Dense(num_classes, kernel_initializer=keras.initializers.he_normal())(out)
	out = keras.layers.Activation('softmax')(out)

	model = keras.models.Model(l_in, out, name='DenseNet121')

	return model