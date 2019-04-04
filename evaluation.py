import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
import model
import input_handler

eval_log = "./eval_log"
checkpoint_dir = "./log"
num_examples = 1000
run_once = False
batch_size = 32


def evaluate():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		
# 	Input loading:
		dataset = input_handler.construct_dataset("test")
		
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
			hooks=[tf.train.StopAtStepHook(last_step=200000)],
			save_checkpoint_secs=None,
			save_checkpoint_steps=None,
			save_summaries_steps=None,
			save_summaries_secs=None,
			config=tf.ConfigProto(
				log_device_placement=False)) as mon_sess:

			mon_sess.run(training_init_op, feed_dict={inputImages: fillerX, inputLabels: fillerY})
			
			totalCount = 0.0
			totalRight = 0.0
			precision = 0.0
			recall = 0.0
			false_pos = 0.0
			true_pos = 0.0
			false_neg = 0.0
			true_neg = 0.0
			while not mon_sess.should_stop():
				try:
					elem = mon_sess.run(next_element, feed_dict={inputImages: fillerX, inputLabels: fillerY})
# 					print("labels:")
# 					print("test:")
# 					print([elem[1]].shape())
# 					print(elem[1].reshape((32,1)).astype(float))
# 					print("output:")
# 					print (elem[0].shape)
					if elem[0].shape[0] >= 32:
						actuals = np.array(elem[1].reshape((32,1)).astype(float), dtype=np.float32)
						predictions = np.round(mon_sess.run(just_sigmoid, feed_dict={inputImages: elem[0], inputLabels: elem[1].reshape((32,1)).astype(float)}))
						print ("Accuracy:")
	# 					print (actuals)
	# 					print (predictions)
						print (float(np.sum(actuals==predictions))/actuals.shape[0] * 100)
						
						
						for p in range(actuals.shape[0]):
							if actuals[p] == predictions[p]:
								if actuals[p] == 1:
									true_pos+=1
								else:
									true_neg+=1
							else:
								if predictions[p] == 0:
									false_neg+=1 
								else:
									false_pos+=1
						
						totalCount += actuals.shape[0]
						totalRight += np.sum(actuals==predictions)
					
				except tf.errors.OutOfRangeError:
					print("End of training dataset. Final Accuracy:")
					print( str(totalRight/totalCount * 100) + "%") 
					precision = true_pos / (true_pos+false_pos)
					recall = true_pos / (true_pos+false_neg)
					print("Precision: " + str(precision) + " Recall: " + str(recall))
					break
				
			
				# print "Logits:"
				
				# print just_softmax.eval(session=mon_sess)

if __name__ == '__main__':
	evaluate()


