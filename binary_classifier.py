from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import fitsio
import numpy
import pickle

from fits_corr import *
from fits_corr.libs.utils import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)



data_loc = "/Users/mulan/desktop/fits_data/"
ext = '.fits'

dataset = []
pairs = []

pair_dict = {} # ON : OFF
label_dict = {} # name : label

sess=tf.InteractiveSession


with open('pairs.pkl', 'rb') as f:
	pair_dict = pickle.load(f)
	# pickle.dump(pair_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/mulan/SETI fall 17/rfi_keras/trainer/labelled_files.pkl', 'rb') as f:
	label_dict = pickle.load(f)

positive_pairs = [] # pairs share different label
negative_pairs = [] # pairs share same label

for on, off in pair_dict.iteritems():
	try:
		on_label = label_dict[on]
		off_label = label_dict[off] 
		if on_label != off_label:
			positive_pairs.append([on, off])
		else:
			negative_pairs.append([on, off])
	except:
		continue

print("pos: " + str(len(positive_pairs)))
print("neg: " + str(len(negative_pairs)))

positive_data = []
negative_data = []

for p in positive_pairs:
	on = fitsio.read(data_loc + str(p[0]) + ext)
	off = fitsio.read(data_loc + str(p[1]) + ext)
	positive_data.append(np.dstack((on, off, on)))

for p in negative_pairs:
	on = fitsio.read(data_loc + str(p[0]) + ext)
	off = fitsio.read(data_loc + str(p[1]) + ext)
	negative_data.append(np.dstack((on, off, on)))

print("length of positive data: " + str(len(positive_data)) + " and individ: " + str(positive_data[0].shape))
print("length of negative data: " + str(len(negative_data)) + " and individ: " + str(negative_data[0].shape))

TRAIN_80_POS = int(len(positive_data) * 0.8)
TRAIN_80_NEG = int(len(negative_data) * 0.8)

# onehot_encoder = OneHotEncoder(sparse=False)

train_positive_data = positive_data[:TRAIN_80_POS]
train_negative_data = negative_data[:TRAIN_80_NEG]
train_label = np.asarray([1] * TRAIN_80_POS + [0] * TRAIN_80_NEG) # 1 is interesting; 0 is not interesting
# train_label = onehot_encoder.fit_transform(train_label.reshape(len(train_label), 1))

eval_positive_data = positive_data[TRAIN_80_POS:]
eval_negative_data = negative_data[TRAIN_80_NEG:]
eval_label = np.asarray([1] * (len(eval_positive_data)) + [0] * (len(eval_negative_data))) # 1 is interesting; 0 is not interesting
# eval_label = onehot_encoder.fit_transform(eval_label.reshape(len(eval_label), 1))

print("length of training labels: " + str(len(train_label)))
print("length of evaluation labels: " + str(len(eval_label)))

def cnn_binary_classifier(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 16, 512, 3]) # ON, OFF, ON

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu)

	#Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1, # 16 * 512 * 32
		pool_size=[2,2],
		strides=2)

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1, # 8 * 256 * 16
		filters=64,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2, # 8 * 256 * 64
		pool_size=[2,2],
		strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 4 * 128 * 64])
	dense = tf.layers.dense(
		inputs=pool2_flat,
		units=1024,
		activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN)

	sess = tf.InteractiveSession()

	# Logits Layer
	logits = tf.layers.dense(
		inputs=dropout, # [batchsize, 1024]
		units=2) # Either same or not

	
	# Add print operation
	logits = tf.Print(logits, [logits], message="This is a: ")



	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  # `logging_hook`.
	  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}


	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	  "accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])
	  }

	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	# Load training and eval data

	train_data = np.asarray(train_positive_data + train_negative_data, dtype=np.float32) # Returns np.array
	train_data = np.asarray(train_data.reshape(TRAIN_80_POS + TRAIN_80_NEG, 16*512*3))
	train_labels = np.asarray(train_label, dtype=np.int32)

	eval_data = np.asarray(eval_positive_data + eval_negative_data, dtype=np.int32) # Returns np.array
	eval_data = tf.cast(eval_data.reshape(8500 - TRAIN_80_POS - TRAIN_80_NEG, 16*512*3), tf.float32)
	eval_labels = np.asarray(eval_label, dtype=np.int32)

	rfi_classifier = tf.estimator.Estimator(
		model_fn=cnn_binary_classifier,
		model_dir="/tmp/model5")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)

	rfi_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = binary_classifier.evaluate(input_fn=eval_input_fn)


	# print(eval_results)

if __name__ == "__main__":
  tf.app.run()








###

## creating label dictionaries

# pairs.extend(filter(lambda(a,b): fitsio.read(a).shape[0] == 16,
#                     get_on_off_pairs(fits_data_loc)))

# pairs = [[p[31:-5] for p in pair] for pair in pairs]
# print(pairs[0])
# pair_dict = {pair[0]: pair[1] for pair in pairs}

## plotting test images

# fig = plt.figure()
# plt.imshow(fitsio.read(data_loc + str(positive_pairs[1][0]) + ext), cmap='gray', aspect = 'auto')
# fig.savefig('test_images/test_pos_0.png', dpi = fig.dpi, transparent = True)
# plt.imshow(fitsio.read(data_loc + str(positive_pairs[1][1]) + ext), cmap='gray', aspect = 'auto')
# fig.savefig('test_images/test_pos_1.png', dpi = fig.dpi, transparent = True)

# fig = plt.figure()
# plt.imshow(fitsio.read(data_loc + str(negative_pairs[0][0]) + ext), cmap='gray', aspect = 'auto')
# fig.savefig('test_images/test_neg_0.png', dpi = fig.dpi, transparent = True)
# fig = plt.figure()
# plt.imshow(fitsio.read(data_loc + str(negative_pairs[0][1]) + ext), cmap='gray', aspect = 'auto')
# fig.savefig('test_images/test_neg_1.png', dpi = fig.dpi, transparent = True)



