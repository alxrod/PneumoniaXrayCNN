# Hellothis is a test comment to alex rodriguez
from datetime import datetime
import time

import tensorflow as tf
import model
import numpy as np
import input_handler

# things i think are going wrong rn
# -something wrong with softmax
# -iterator only going through once?

def train():

	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		dataset = input_handler.construct_dataset("train")
		dataset = dataset.repeat()
		# makes sure that the dataset keeps feeding into the model on a loop 

		# print dataset
		iterator = tf.data.Iterator.from_structure(dataset.output_types,
												   dataset.output_shapes)
		next_element = iterator.get_next()
		training_init_op = iterator.make_initializer(dataset)

		fillerX = np.zeros((model.BATCH_SIZE,64,64,1)).astype(float)
		fillerY = np.zeros((model.BATCH_SIZE,1)).astype(float)

		inputLabels = tf.placeholder(tf.float32, shape=(model.BATCH_SIZE,1), name="inputLabels")
		inputImages = tf.placeholder(tf.float32, shape=(model.BATCH_SIZE,64,64,1), name="inputLabels")
		
		logits = model.model(inputImages)
		just_sigmoid = tf.nn.sigmoid(logits)
		loss = model.loss(logits, inputLabels)
		train_op = model.train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				log_frequency = 10
				# print "trying to log after"
				if self._step % log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time
					loss_value = run_values.results
					examples_per_sec = log_frequency * model.BATCH_SIZE / duration
					sec_per_batch = float(duration / log_frequency)
					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
				                        'sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value,
				               examples_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
# 		This will pick up where it left off!
# 		If you need to restart, empty the log directory
			checkpoint_dir="./log",
			hooks=[tf.train.StopAtStepHook(last_step=100000),
				   tf.train.NanTensorHook(loss),
				   _LoggerHook()],
			save_checkpoint_steps=200,
			config=tf.ConfigProto(
				log_device_placement=False)) as mon_sess:

			mon_sess.run(training_init_op, feed_dict={inputImages: fillerX, inputLabels: fillerY})
			while not mon_sess.should_stop():
				try:
					elem = mon_sess.run(next_element, feed_dict={inputImages: fillerX, inputLabels: fillerY})
# 					print("labels:")
# 					print("test:")
# 					print([elem[1]].shape())
# 					print(elem[1].reshape((32,1)).astype(float))
# 					print("output:")
					actuals = np.array(elem[1].reshape((32,1)).astype(float), dtype=np.float32)
					predictions = np.round(mon_sess.run(just_sigmoid, feed_dict={inputImages: elem[0], inputLabels: elem[1].reshape((32,1)).astype(float)}))
					print ("Accuracy:")
# 					print (actuals)
# 					print (predictions)
					print (float(np.sum(actuals==predictions))/actuals.shape[0] * 100)
					mon_sess.run(train_op, feed_dict={inputImages: elem[0], inputLabels: elem[1].reshape((32,1)).astype(float)})
					
# 					print(mon_sess.run(just_softmax, feed_dict={inputImages: elem[0], inputLabels: elem[1]}))
				except tf.errors.OutOfRangeError:
					mon_sess.run(training_init_op, feed_dict={inputImages: fillerX, inputLabels: fillerY})
					print("End of training dataset.")
				
			
				# print "Logits:"
				
				# print just_softmax.eval(session=mon_sess)

if __name__ == '__main__':
	train()

