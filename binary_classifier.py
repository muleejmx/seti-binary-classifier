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

# tf.reset_default_graph()

# # input images
# with tf.name_scope('input'):
#     # None -> batch size can be any size, 784 -> flattened mnist image
#     x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
#     # target 10 output classes
#     y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

# # model parameters will change during training so we use tf.Variable
# with tf.name_scope("weights"):
#     W = tf.Variable(tf.zeros([784, 10]))

# # bias
# with tf.name_scope("biases"):
#     b = tf.Variable(tf.zeros([10]))

# # implement model
# with tf.name_scope("softmax"):
#     # y is our prediction
#     y = tf.nn.softmax(tf.matmul(x,W) + b)

# # specify cost function
# with tf.name_scope('cross_entropy'):
#     # this is our cost
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# # specify optimizer
# with tf.name_scope('train'):
#     # optimizer is an "operation" which we can execute in a session
#     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# with tf.name_scope('Accuracy'):
#     # Accuracy
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# # create a summary for our cost and accuracy
# tf.scalar_summary("cost", cross_entropy)
# tf.scalar_summary("accuracy", accuracy)

# # merge all summaries into a single "operation" which we can execute in a session 
# summary_op = tf.merge_all_summaries()

###

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

      on_sorted = np.sort(on_data)[int(0.25*len(on_data)):int(0.75*len(on_data))]
      off_sorted = np.sort(off_data)[int(0.25*len(off_data)):int(0.75*len(off_data))]

      on_avg = np.mean(on_sorted)
      off_avg = np.mean(off_sorted)

      on_std = np.std(on_sorted)
      off_std = np.std(off_sorted)

      on_data[on_data < (on_avg + 10*on_std)] = 0
      on_data[on_data > (on_avg + 10*on_std)] = 1

      off_data[off_data < (off_avg + 10*off_std)] = 0
      off_data[off_data > (off_avg + 10*off_std)] = 1

      on_flip_ud = np.flipud(on_data)
      on_flip_lr = np.fliplr(on_data)
      
      off_flip_ud = np.flipud(off_data)
      off_flip_lr = np.fliplr(off_data)

      roll_val = np.random.randint(-25, 26, 6)

      # Add Data

      stack(on_data, off_data, typ) # 1
      stack(on_flip_ud, off_flip_ud, typ) # 2
      stack(on_flip_lr, off_flip_lr, typ) # 3

      stack(np.roll(on_data, roll_val[0]), np.roll(off_data, roll_val[1]), typ) # 4
      stack(np.roll(on_flip_ud, roll_val[2]), np.roll(off_flip_ud, roll_val[3]), typ) # 5
      stack(np.roll(on_flip_lr, roll_val[4]), np.roll(off_flip_lr, roll_val[5]), typ) # 6

      if int(on_label) == 6 or int(on_label) == 7 or int(off_label) == 6 or int(off_label) == 7:
        stack(on_data, off_flip_lr, True)
        stack(on_flip_lr, off_data, True)
        stack(on_flip_ud, off_data, True)
        stack(on_data, off_flip_ud, True)
	
	roll_val = np.random.randint(-25, 26, 8)
	stack(np.roll(on_data, roll_val[0]), np.roll(off_flip_lr, roll_val[1]), True)
        stack(np.roll(on_flip_lr, roll_val[2]), np.roll(off_data, roll_val[3]), True)
        stack(np.roll(on_flip_ud, roll_val[4]), np.roll(off_data, roll_val[5]), True)
        stack(np.roll(on_data, roll_val[6]), np.roll(off_flip_ud, roll_val[7]), True)


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

eval_data = np.asarray(eval_positive_data + eval_negative_data, dtype=np.int32) # Returns np.array
eval_data = np.asarray(eval_data.reshape(len(positive_data) + len(negative_data) - TRAIN_80_POS - TRAIN_80_NEG, 16*512*2), dtype=np.int32)
eval_labels = np.asarray(eval_label, dtype=np.int32)

print(len(eval_data))
print(len(eval_labels))

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 16, 512, 2])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
      inputs=conv1, # 16 * 512 * 32
      pool_size=[2, 4],
      strides=[2, 4]) 

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1, # 8 * 128 * 16
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
      inputs=conv2, # 8 * 128 * 64
      pool_size=[2, 4],
      strides=[2, 4])

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 4 * 32 * 64])

    dense = tf.layers.dense(
      inputs=pool2_flat,
      units=1024,
      activation=tf.nn.relu)

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

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels,
          predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    train_data = np.asarray(train_positive_data + train_negative_data, dtype=np.float32)
    train_data = np.asarray(train_data.reshape(TRAIN_80_POS + TRAIN_80_NEG, 16*512*2))
    train_labels = np.asarray(train_label, dtype=np.int32)

    eval_data = np.asarray(eval_positive_data + eval_negative_data, dtype=np.float32) # Returns np.array
    eval_data = np.asarray(eval_data.reshape(len(positive_data) + len(negative_data) - TRAIN_80_POS - TRAIN_80_NEG, 16*512*2))
    eval_labels = np.asarray(eval_label, dtype=np.int32)

    # Create the Estimator
    rfi_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="./tmp/05")

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
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    rfi_classifier.train(
        input_fn=train_input_fn,
        steps=40000,
        hooks=[logging_hook])
    print("length of eval_data: " + str(len(eval_data)))
    print("length of eval_labels: " + str(len(eval_labels)))
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data[:10000]},
        y=eval_labels[:10000],
        num_epochs=1,
        shuffle=False)

    eval_results = rfi_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

if __name__ == "__main__":
    tf.app.run()






