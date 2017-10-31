from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import fitsio
import numpy
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

ext = '.fits'
data_loc = './fits_data/'
dataset = [] # name of on pairs
positive_data = [] # pairs share different label
negative_data = [] # pairs share same label

with open('./data/labelled_files.pkl', 'rb') as f:
    label_dict = pickle.load(f) # name:label

counter = 0
for k, v in label_dict.iteritems():
  if k[-3:] != 'OFF': # ON
    dataset.append(k)
print("num of pairs: " + str(len(dataset)))


# Our application logic will be added here

def stack(dataON, dataOFF, typ):
  """ Takes in data (as outputted by fitsio read)
  and npstacks it. 
  typ = True if positive (different labels)
  typ = False if negative (same labels) """
  if typ:
    positive_data.append(np.dstack((dataON, dataOFF)))
  else:
    negative_data.append(np.dstack((dataON, dataOFF)))

def training_data():

  for on in dataset:
    off = on + '_OFF'

    try:

      on_label = label_dict[on]
      off_label = label_dict[off] 

      typ = (on_label != off_label)

      on_data = fitsio.read(data_loc + str(on) + ext)
      off_data = fitsio.read(data_loc + str(off) + ext)

      on_min = min(on_data.flatten())
      on_max = max(on_data.flatten())

      off_min = min(off_data.flatten())
      off_max = max(off_data.flatten())

      on_data = np.round((on_data - on_min)*(255/(on_max-on_min)))

      off_data = np.round((off_data - off_min)* (255/(off_max-off_min)))
      # on_sorted = np.sort(on_data)[int(0.25*len(on_data)):int(0.75*len(on_data))]
      # off_sorted = np.sort(off_data)[int(0.25*len(off_data)):int(0.75*len(off_data))]

      # on_avg = np.mean(on_sorted)
      # off_avg = np.mean(off_sorted)

      # on_std = np.std(on_sorted)
      # off_std = np.std(off_sorted)

      # on_data[on_data < (on_avg + 10*on_std)] = 0
      # on_data[on_data > (on_avg + 10*on_std)] = 1

      # off_data[off_data < (off_avg + 10*off_std)] = 0
      # off_data[off_data > (off_avg + 10*off_std)] = 1

      on_flip_ud = np.flipud(on_data)
      on_flip_lr = np.fliplr(on_data)
      
      off_flip_ud = np.flipud(off_data)
      off_flip_lr = np.fliplr(off_data)

      on_db_flip = np.flipud(on_flip_lr)
      off_db_flip = np.flipud(off_flip_lr)

      roll_val = np.random.randint(-25, 26, 20)

      # Add Data

      stack(on_data, off_data, typ) # 1
      stack(np.roll(on_data, roll_val[0]), np.roll(off_data, roll_val[0]), typ)
      stack(np.roll(on_data, roll_val[1]), np.roll(off_data, roll_val[1]), typ)
      stack(np.roll(on_data, roll_val[2]), np.roll(off_data, roll_val[2]), typ)
      #stack(np.roll(on_data, roll_val[3]), np.roll(off_data, roll_val[3]), typ)
      #stack(np.roll(on_data, roll_val[4]), np.roll(off_data, roll_val[4]), typ)

      stack(on_flip_ud, off_flip_ud, typ) # 2
      stack(np.roll(on_flip_ud, roll_val[5]), np.roll(off_flip_ud, roll_val[5]), typ)
      stack(np.roll(on_flip_ud, roll_val[6]), np.roll(off_flip_ud, roll_val[6]), typ)
      stack(np.roll(on_flip_ud, roll_val[7]), np.roll(off_flip_ud, roll_val[7]), typ)
      #stack(np.roll(on_flip_ud, roll_val[8]), np.roll(off_flip_ud, roll_val[8]), typ)
      #stack(np.roll(on_flip_ud, roll_val[9]), np.roll(off_flip_ud, roll_val[9]), typ)

      stack(on_flip_lr, off_flip_lr, typ) # 3
      stack(np.roll(on_flip_lr, roll_val[10]), np.roll(off_flip_lr, roll_val[10]), typ)
      stack(np.roll(on_flip_lr, roll_val[11]), np.roll(off_flip_lr, roll_val[11]), typ)
      stack(np.roll(on_flip_lr, roll_val[12]), np.roll(off_flip_lr, roll_val[12]), typ)
      #stack(np.roll(on_flip_lr, roll_val[13]), np.roll(off_flip_lr, roll_val[13]), typ)
      #stack(np.roll(on_flip_lr, roll_val[14]), np.roll(off_flip_lr, roll_val[14]), typ)

      stack(on_db_flip, off_db_flip, typ) # 4
      stack(np.roll(on_db_flip, roll_val[15]), np.roll(off_db_flip, roll_val[15]), typ)
      stack(np.roll(on_db_flip, roll_val[16]), np.roll(off_db_flip, roll_val[16]), typ)
      stack(np.roll(on_db_flip, roll_val[17]), np.roll(off_db_flip, roll_val[17]), typ)
      # stack(np.roll(on_db_flip, roll_val[18]), np.roll(off_db_flip, roll_val[18]), typ)
      # stack(np.roll(on_db_flip, roll_val[19]), np.roll(off_db_flip, roll_val[19]), typ)

    except "FileNotFoundError":
      pass


training_data()
print("length of positive data: " + str(len(positive_data)) + " shape: " + str(len(positive_data[0])))
print("length of negative data: " + str(len(negative_data)) + " shape: " + str(len(negative_data[0])))

TRAIN_80_POS = int(len(positive_data) * 0.8)
TRAIN_80_NEG = int(len(negative_data) * 0.8)

np.random.shuffle(positive_data)
np.random.shuffle(negative_data)

train_positive_data = positive_data[:TRAIN_80_POS]
train_negative_data = negative_data[:TRAIN_80_NEG]

eval_positive_data = positive_data[TRAIN_80_POS:]
eval_negative_data = negative_data[TRAIN_80_NEG:]

train_label = [1] * TRAIN_80_POS + [0] * TRAIN_80_NEG
eval_label = [1] * (len(positive_data) - TRAIN_80_POS) + [0] * (len(negative_data) - TRAIN_80_NEG)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 16, 512, 2])
    #tf.summary.image('input', tf.reshape(features["x"], [-1, 16, 512, 2]), 2)

    # Convolutional Layer #1
    with tf.name_scope('conv1'):
      conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1')
      conv1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1')
      tf.summary.histogram('kernel', conv1_vars[0])
      tf.summary.histogram('bias', conv1_vars[1])
      tf.summary.histogram('act', conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
      inputs=conv1, # 16 * 512 * 32
      pool_size=[2, 4],
      strides=[2, 4],
      name='pool1') 

    # Convolutional Layer #2 and Pooling Layer #2
    with tf.name_scope('conv2'):
      conv2 = tf.layers.conv2d(
        inputs=pool1, # 8 * 128 * 16
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2')
      conv2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv2')
      tf.summary.histogram('kernel', conv2_vars[0])
      tf.summary.histogram('bias', conv2_vars[1])
      tf.summary.histogram('act', conv2)

    pool2 = tf.layers.max_pooling2d(
      inputs=conv2, # 8 * 128 * 64
      pool_size=[2, 4],
      strides=[2, 4],
      name='pool2')

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 4 * 32 * 64])

    with tf.name_scope('dense'):
      dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu,
        name='dense')
      dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense')
      tf.summary.histogram('kernel', dense_vars[0])
      tf.summary.histogram('bias', dense_vars[1])
      tf.summary.histogram('act', dense)

    dropout = tf.layers.dropout(
      inputs=dense,
      rate=0.4,
      training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    tf.Print(tf.nn.softmax(logits), [tf.reduce_min(logits)], message="recorded probabilities")

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    with tf.name_scope('loss'):
      unregularized_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
      #l2_loss = tf.nn.l2_loss(conv1)+tf.nn.l2_loss(conv2)+tf.nn.l2_loss(dense)
      #loss = tf.add(unregularized_loss, l2_loss*0.002)
      loss = unregularized_loss
      tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.name_scope('train_op'):
        global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.0008
	k = 0.5
	
	#natural exp decay; 14
	learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step, 1000, k)
	# inverse time decay; 13
	#learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, 10000, k)
	# exponential decay; 12
	#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
       
	train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels,
          predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
          labels=labels,
          predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
          labels=labels,
          predictions=predictions["classes"]),
      "false_positives": tf.contrib.metrics.streaming_false_positives(
	  labels=labels,
	  predictions=predictions["classes"]),
      "true_positives": tf.contrib.metrics.streaming_true_positives(
          labels=labels,
          predictions=predictions["classes"]),
      "false_negatives": tf.contrib.metrics.streaming_false_negatives(
          labels=labels,
          predictions=predictions["classes"]),
	"true_negatives": tf.contrib.metrics.streaming_true_negatives(
          labels=labels,
          predictions=predictions["classes"]),
}

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    ########################
    train_data = np.asarray((train_positive_data + train_negative_data), dtype=np.float32)
    train_data = np.asarray(train_data.reshape(TRAIN_80_POS + TRAIN_80_NEG, 16*512*2))
    train_labels = np.asarray(train_label, dtype=np.int32)

    eval_data = np.asarray(eval_positive_data + eval_negative_data, dtype=np.float32) # Returns np.array
    eval_data = np.asarray(eval_data.reshape(len(eval_data), 16*512*2))
    eval_labels = np.asarray(eval_label, dtype=np.int32)

    # Create the Estimator
    rfi_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="./tmp/29")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=50)

    #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=16,
        num_epochs=None,
        shuffle=True)

    rfi_classifier.train(
        input_fn=train_input_fn,
        steps=100000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = rfi_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

if __name__ == "__main__":
    tf.app.run()






