import os

# Limit tensorflow logs to warning or errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import math
import random
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from imagenet_functions import create_vgg16, create_densenet121, load_imagenet, load_imagenet224, compute_metrics, print_metrics

# Sacred
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

# Sacred Configuration
ex = Experiment('AF-Imagenet224')
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS.CAPTURE_MODE = 'no'  # Do not keep track of prints and stdout, stderr and so on.
ex.observers.append(MongoObserver(url="mongodb://david_mongo:david_mongo@192.168.117.142:27017/?authMechanism=SCRAM-SHA-1", db_name='db'))


@ex.config
def cfg():
	# seeds
	seed = 1

	# Activation function used in the model
	activation = 'elusppt'

	optimiser_params = {
		# Loss function for optimiser
		'loss' : 'sparse_categorical_crossentropy',

		# Metrics evaluated during training
		'metrics' : ['accuracy'],

		# Learning rate decay for optimiser
		'lr_decay' : 1e-6,
		
		# Initial learning rate for optimiser
		'lr' : 0.1
	}
	
	# Batch size for train and validation/test
	batch_size = 320
	val_batch_size = 320

	# Number of epochs
	epochs = 100

	# Number of train batches to load from imagenet
	# -1 means all
	num_train_batches = -1

	# Type of model architecture
	model_name = 'vgg'

@ex.automain
@LogFileWriter(ex)
def train_and_evaluate(seed, activation, optimiser_params, batch_size, val_batch_size, epochs, num_train_batches, model_name):
	# Set mixed precision policy to use float16
	policy =  mixed_precision.Policy('mixed_float16')
	mixed_precision.set_global_policy(policy)

	# Create distributed training strategy
	mirrored_strategy = tf.distribute.MirroredStrategy()

	# Set dataset details
	num_classes = 1000
	shape = (224, 224, 3)

	# Create model with distributed strategy
	with mirrored_strategy.scope():
		if model_name == "vgg":
			model = create_vgg16(shape, num_classes, activation)
		elif model_name == "densenet":
			model = create_densenet121(shape, num_classes, activation)
		else:
			raise Exception("Model name is not valid!")


		# Compile model with desired optimizer, loss and metrics.
		model.compile(
			optimizer = keras.optimizers.SGD(lr=optimiser_params['lr'], decay=optimiser_params['lr_decay'], momentum=0.9, nesterov=True),
			loss=optimiser_params['loss'], metrics=optimiser_params['metrics']
		)

	# Print model summary
	model.summary()

	# Create TF dataset
	def train_generator():
		for i, j in zip(train_x, train_y):
			yield i, j

	def data_augmentation(x, y):
		if random.random() > 0.5:
			x = tf.image.flip_left_right(x)
		if random.random() > 0.5:
			x = tf.keras.preprocessing.image.random_shift(x, random.random() / 10.0, random.random() / 10.0, channel_axis=3)
		if random.random() > 0.5:
			x = tf.keras.preprocessing.image.random_rotation(x, random.random() * 20.0, channel_axis=3)

		return x, y

	train_ds, val_ds, test_ds, train_size, val_size, test_size = load_imagenet224(num_train_batches, 0.05, seed)
	train_ds = train_ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_ds = val_ds.batch(val_batch_size).prefetch(tf.data.AUTOTUNE)
	test_ds = test_ds.batch(val_batch_size).prefetch(tf.data.AUTOTUNE)

	test_y = []
	for example in test_ds:
		test_y = [*test_y, *list(example[1].numpy())]

	# Set auto sharding to DATA
	ds_options = tf.data.Options()
	ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	train_ds = train_ds.with_options(ds_options)
	val_ds = val_ds.with_options(ds_options)
	test_ds = test_ds.with_options(ds_options)

	# Distribute datasets
	train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
	val_ds = mirrored_strategy.experimental_distribute_dataset(val_ds)
	test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)

	print(f'Training on {train_size} samples and validating on {val_size} samples.')

	def log_metrics(epoch, logs):
		ex.log_scalar("train_loss", logs['loss'])
		ex.log_scalar("train_acc", logs['accuracy'])
		ex.log_scalar("val_loss", logs['val_loss'])
		ex.log_scalar("val_accuracy", logs['val_accuracy'])

	steps_train = math.ceil(train_size / batch_size)
	steps_val = math.ceil(val_size / val_batch_size)
	steps_test = math.ceil(test_size / val_batch_size)

	# Train model
	model.fit(
		train_ds,
		epochs=epochs,
		steps_per_epoch=steps_train,
		callbacks=[
				keras.callbacks.LambdaCallback(on_epoch_end=log_metrics),
				keras.callbacks.LearningRateScheduler(lambda epoch: optimiser_params['lr'] * (0.5 ** (epoch // 20))),
				keras.callbacks.EarlyStopping(min_delta=0.0005, patience=40, verbose=1, restore_best_weights=True)
		    ],
		validation_data=val_ds,
		validation_steps=steps_val,
		verbose=1
	)

	# Evaluate the model on test set
	predictions = model.predict(test_ds, steps=steps_test)
	metrics = compute_metrics(test_y, predictions, num_classes)
	print_metrics(metrics)

	# loss, accuracy = model.evaluate(test_ds, steps=steps_test)
	# metrics = {'loss' : loss, 'CCR' : accuracy}

	# Choose unique directory to avoid errors when running multiple experiments at the same time
	temp_dir = Path(f"./temp{random.randint(1000000,999999999)}/")
	while temp_dir.is_dir():
		temp_dir = Path(f"./temp{random.randint(1000000, 999999999)}/")
	os.mkdir(temp_dir)
	# Creates files with metrics and confusion matrix to add them as artifacts for sacred.
	with open(temp_dir / 'metrics.csv', 'w') as f:
		for key in metrics.keys():
			if key is not 'ConfMat':
				f.write("%s,%s\n" % (key, metrics[key]))

	ex.add_artifact(temp_dir / 'metrics.csv')

	shutil.rmtree(temp_dir)

	return float(metrics['CCR'])
