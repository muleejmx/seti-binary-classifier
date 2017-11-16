from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Imports
import numpy as np
import tensorflow as tf
import fitsio
import matplotlib.pyplot as plt
import numpy
import pickle
import scipy.stats

from StringIO import StringIO


from tensorflow.python.training.session_run_hook import SessionRunArgs

class Logger(object):
  """Logging in tensorboard without tensorflow ops."""

  def __init__(self, log_dir):
      """Creates a summary writer logging to log_dir."""
      self.writer = tf.summary.FileWriter(log_dir)

  def log_scalar(self, tag, value, step):
      """Log a scalar variable.
      Parameter
      ----------
      tag : basestring
          Name of the scalar
      value
      step : int
          training iteration
      """
      summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                   simple_value=value)])
      self.writer.add_summary(summary, step)

  def log_images(self, tag, images, step):
      """Logs a list of images."""

      im_summaries = []
      for nr, img in enumerate(images):
          # Write the image to a string
          s = StringIO()
          plt.imsave(s, img, format='png')

          # Create an Image object
          img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                     height=img.shape[0],
                                     width=img.shape[1])
          # Create a Summary value
          im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                               image=img_sum))

      # Create and write Summary
      summary = tf.Summary(value=im_summaries)
      self.writer.add_summary(summary, step)
      

  def log_histogram(self, tag, values, step, bins=1000):
      """Logs the histogram of a list/vector of values."""
      # Convert to a numpy array
      values = np.array(values)
      
      # Create histogram using numpy        
      counts, bin_edges = np.histogram(values, bins=bins)

      # Fill fields of histogram proto
      hist = tf.HistogramProto()
      hist.min = float(np.min(values))
      hist.max = float(np.max(values))
      hist.num = int(np.prod(values.shape))
      hist.sum = float(np.sum(values))
      hist.sum_squares = float(np.sum(values**2))

      # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
      # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
      # Thus, we drop the start of the first bin
      bin_edges = bin_edges[1:]

      # Add bin edges and counts
      for edge in bin_edges:
          hist.bucket_limit.append(edge)
      for c in counts:
          hist.bucket.append(c)

      # Create and write Summary
      summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
      self.writer.add_summary(summary, step)
      self.writer.flush()


tf.logging.set_verbosity(tf.logging.INFO)

ext = '.fits'
data_loc = './fits_data/'
dataset = [] # name of ON pairs
allData = []

logger = Logger('./tmp/testHist')

with open('./data/labelled_files.pkl', 'rb') as f:
    label_dict = pickle.load(f) # name:label

counter = 0
for k, v in label_dict.iteritems():
  if k[-3:] != 'OFF': # ON
    dataset.append(k)

print("num of pairs: " + str(len(dataset)))

# Our application logic will be added here

def stack(dataON, dataOFF, typ, label):
  """ Takes in data (as outputted by fitsio read)
  and npstacks it. 
  typ = 1-10 if same labels (based on ON)
  typ = 10-20 if different labels """
  allData.append([np.dstack((dataON, dataOFF)), typ, label])

def training_data():

  def normalizerIQR(onArr, offArr):
    """Normalize an ON/OFF pair using median removal and IQR scaling"""
    
    # saturate signals above and below the IQR centred on the median
    # use the data with the max absolute value for median and IQR for this
    # operation, this assumes the on and off data are similarly scaled

    if np.abs(onArr).max() > np.abs(offArr).max():
      median = np.median(onArr)
      IQR = scipy.stats.iqr(onArr)
    else:
      median = np.median(offArr)
      IQR = scipy.stats.iqr(offArr)

    # median subtract and normalize to range [-1,1]
    onArrNorm = (np.clip(onArr, median - IQR/2., median + IQR/2.) - median) / (IQR/2.)
    offArrNorm = (np.clip(offArr, median - IQR/2., median + IQR/2.) - median) / (IQR/2.)

    return onArrNorm, offArrNorm

  def normalizer8bit(onArr, offArr):
    """Normalize an ON/OFF pair to 8-bits, rescale to [0,1]"""
    on_min = onArr.min()
    on_max = onArr.max()

    off_min = offArr.min()
    off_max = offArr.max()

    onArrNorm = np.round(256. * ((onArr - on_min) / (on_max - on_min))) / 256.
    offArrNorm = np.round(256. * ((offArr - off_min) / (off_max - off_min))) / 256.

    return onArrNorm, offArrNorm

  for on in dataset:
    off = on + '_OFF'

    try:

      on_label = label_dict[on]
      off_label = label_dict[off] 


      if (on_label >= 8 and on_label <= 11) or (off_label >= 8 and off_label <= 11) or on_label == 14 or off_label == 14: # not enough samples
        continue
      else:
        if on_label ==  12 or on_label == 13:
          on_label -= 4
        if off_label == 12 or off_label == 13:
          off_label -= 4

      if (on_label != off_label):
        typ = 10+on_label
      else:
        typ = on_label

      on_data_raw = fitsio.read(data_loc + str(on) + ext)
      off_data_raw = fitsio.read(data_loc + str(off) + ext)

      #on_data, off_data = normalizer8bit(on_data_raw, off_data_raw)
      on_data, off_data = normalizerIQR(on_data_raw, off_data_raw)

      on_flip_ud = np.flipud(on_data)
      on_flip_lr = np.fliplr(on_data)
      
      off_flip_ud = np.flipud(off_data)
      off_flip_lr = np.fliplr(off_data)

      on_db_flip = np.flipud(on_flip_lr)
      off_db_flip = np.flipud(off_flip_lr)

      roll_val = np.random.randint(-25, 26, 20)

      # Add Data

      stack(on_data, off_data, typ, str(on)) # 1
      stack(np.roll(on_data, roll_val[0]), np.roll(off_data, roll_val[0]), typ, str(on))
      stack(np.roll(on_data, roll_val[1]), np.roll(off_data, roll_val[1]), typ, str(on))
      stack(np.roll(on_data, roll_val[2]), np.roll(off_data, roll_val[2]), typ, str(on))
      #stack(np.roll(on_data, roll_val[3]), np.roll(off_data, roll_val[3]), typ)
      #stack(np.roll(on_data, roll_val[4]), np.roll(off_data, roll_val[4]), typ)

      stack(on_flip_ud, off_flip_ud, typ, str(on)) # 2
      stack(np.roll(on_flip_ud, roll_val[5]), np.roll(off_flip_ud, roll_val[5]), typ, str(on))
      stack(np.roll(on_flip_ud, roll_val[6]), np.roll(off_flip_ud, roll_val[6]), typ, str(on))
      stack(np.roll(on_flip_ud, roll_val[7]), np.roll(off_flip_ud, roll_val[7]), typ, str(on))
      #stack(np.roll(on_flip_ud, roll_val[8]), np.roll(off_flip_ud, roll_val[8]), typ)
      #stack(np.roll(on_flip_ud, roll_val[9]), np.roll(off_flip_ud, roll_val[9]), typ)

      stack(on_flip_lr, off_flip_lr, typ, str(on)) # 3
      stack(np.roll(on_flip_lr, roll_val[10]), np.roll(off_flip_lr, roll_val[10]), typ, str(on))
      stack(np.roll(on_flip_lr, roll_val[11]), np.roll(off_flip_lr, roll_val[11]), typ, str(on))
      stack(np.roll(on_flip_lr, roll_val[12]), np.roll(off_flip_lr, roll_val[12]), typ, str(on))
      #stack(np.roll(on_flip_lr, roll_val[13]), np.roll(off_flip_lr, roll_val[13]), typ)
      #stack(np.roll(on_flip_lr, roll_val[14]), np.roll(off_flip_lr, roll_val[14]), typ)

      stack(on_db_flip, off_db_flip, typ, str(on)) # 4
      stack(np.roll(on_db_flip, roll_val[15]), np.roll(off_db_flip, roll_val[15]), typ, str(on))
      stack(np.roll(on_db_flip, roll_val[16]), np.roll(off_db_flip, roll_val[16]), typ, str(on))
      stack(np.roll(on_db_flip, roll_val[17]), np.roll(off_db_flip, roll_val[17]), typ, str(on))
      # stack(np.roll(on_db_flip, roll_val[18]), np.roll(off_db_flip, roll_val[18]), typ)
      # stack(np.roll(on_db_flip, roll_val[19]), np.roll(off_db_flip, roll_val[19]), typ)

    except "FileNotFoundError":
      pass


training_data()

TRAIN_80 = int(len(allData) * 0.8)

np.random.shuffle(allData)

trainData = allData[:TRAIN_80]
evalData = allData[TRAIN_80:]

print("length of data: " + str(len(allData)))
print("length of training data: " + str(len(trainData)))
print("length of eval data: " + str(len(evalData)))

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
    logits = tf.layers.dense(inputs=dropout, units=20)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=20)
    with tf.name_scope('loss'):
      unregularized_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
      #loss = tf.add(unregularized_loss, l2_loss*0.002)
      loss = unregularized_loss
      tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.name_scope('train_op'):
        global_step = tf.Variable(0, trainable=False)

	starter_learning_rate = 0.0005
	k = 0.5
	
	learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step, 10000, k)
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
          predictions=predictions["classes"])
    }   
    
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    ########################
    train_data = np.asarray([t[0] for t in trainData], dtype=np.float32)
    train_data = np.asarray(train_data.reshape(len(train_data), 16*512*2))
    train_labels = np.asarray([t[1] for t in trainData], dtype=np.int32)

    eval_data = np.asarray([t[0] for t in evalData], dtype=np.float32) # Returns np.array
    eval_data = np.asarray(eval_data.reshape(len(eval_data), 16*512*2))
    eval_labels = np.asarray([t[1] for t in evalData], dtype=np.int32)

    eval_names = np.asarray([t[2] for t in evalData])

    # Create the Estimator
    rfi_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="./tmp/05")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    # logging_hook = tf.train.LoggingTensorHook(
    #   tensors=tensors_to_log,
    #   every_n_iter=50)

    eval_logging_hook = EvalLoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=1)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    rfi_classifier.train(
        input_fn=train_input_fn,
        steps=40000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
	batch_size=32,
        shuffle=False)

    eval_results = rfi_classifier.evaluate(
      input_fn=eval_input_fn,
      hooks=[eval_logging_hook])

    numpy.savetxt("./eval_labels.csv", eval_labels.astype(np.int64), delimiter=",")
   # numpy.savetxt("./eval_names.csv", eval_names, delimiter=",")
    print("Saved" + str(eval_results))
    #print("Saved" + str(eval_names))

class EvalLoggingTensorHook(tf.train.LoggingTensorHook):
  """A revised version of LoggingTensorHook to use during evaluation.
  This version supports being reset and increments `_iter_count` before run
  instead of after run.
  """

  def begin(self):
    # Reset timer.
    self._timer.update_last_triggered_step(0)
    super(EvalLoggingTensorHook, self).begin()

  def before_run(self, run_context):
    self._iter_count += 1
    return super(EvalLoggingTensorHook, self).before_run(run_context)

  def after_run(self, run_context, run_values):
    super(EvalLoggingTensorHook, self).after_run(run_context, run_values)
    self._iter_count -= 1


if __name__ == "__main__":
    tf.app.run()

