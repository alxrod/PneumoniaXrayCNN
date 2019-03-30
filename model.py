import tensorflow as tf

IMAGE_SIZE = (64,64)
NUM_CLASSES = 2



NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999     
NUM_EPOCHS_PER_DECAY = 350.0  
LEARNING_RATE_DECAY_FACTOR = 0.1 
INITIAL_LEARNING_RATE = 0.1  

BATCH_SIZE = 32 

def _makeVariable(name, shape, initializer):
	var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return var

def _makeRandomizedVariable(name, shape, stddev):
	var = _makeVariable(name, shape, tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32))
	return var

def model(images):
	with tf.variable_scope('conv1') as scope:
		kernel = _makeRandomizedVariable('weights',
										 shape=[5,5,1,64],
										 stddev=5e-2)
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
		biases = _makeVariable("biases", [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pool1")

	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

	with tf.variable_scope('conv2') as scope:
		kernel = _makeRandomizedVariable('weights',
										 shape=[5,5,64,64],
										 stddev=5e-2)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding="SAME")
		biases = _makeVariable("biases", [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

	pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pool2")

	# fully connected layer here
	with tf.variable_scope("local3") as scope:
		# flatten to a 1d imagesize by batch size
		reshape = tf.reshape(pool2, [list(images.shape)[0], -1])
		# numbner of batches
		dim = reshape.get_shape()[1].value
		weights = _makeRandomizedVariable('weights', shape=[dim,384], stddev=0.04)
		biases = _makeVariable("biases", [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)

	with tf.variable_scope("local4") as scope:
		weights = _makeRandomizedVariable("weights", shape=[384,192],stddev=0.04)
		biases = _makeVariable("biases", [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

	with tf.variable_scope("softmax_linear") as scope:
		weights = _makeRandomizedVariable("weights", [192, NUM_CLASSES], stddev=1/192.0)
		biases = _makeVariable("biases", [NUM_CLASSES], tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4,weights), biases, name=scope.name)

	return softmax_linear

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name="cross_entropy_per_example")
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
	tf.add_to_collection("losses",cross_entropy_mean)

	# add_n is element wise adding
	return tf.add_n(tf.get_collection("losses"), name="total_loss")

def _add_loss(total_loss):
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	loss_averages_op = loss_averages.apply([total_loss])
	for l in [total_loss]:
	    tf.summary.scalar(l.op.name + ' (raw)', l)
	    tf.summary.scalar(l.op.name, loss_averages.average(l))
	return loss_averages_op


def train(total_loss, global_step):
	num_batches_per_epoc = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 32
	decay_steps = int(num_batches_per_epoc * NUM_EPOCHS_PER_DECAY)

	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
									global_step,
									decay_steps,
									LEARNING_RATE_DECAY_FACTOR,
									staircase=True)

	tf.summary.scalar('learning_rate', lr)
	loss_averages_op = _add_loss(total_loss)

	# control dependencies means make sure the loss average op has been evaluated at this point
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

	return apply_gradients_op





		