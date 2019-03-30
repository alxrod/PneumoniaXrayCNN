import tensorflow as tf
import random
import os
tf.enable_eager_execution()

def getFileNames(directory):
	nameList = os.listdir(directory)
	imageNames = []
	for f in nameList:
		if "resized" in f:
			imageNames.append(f)

	random.shuffle(imageNames)

	filePaths = []
	for f in imageNames:
	    filePaths.append(directory+f)

	return filePaths

def getLabels(filePaths):
	label_names = ["pneumonia", "normal"]
	label_to_index = dict((name, index) for index,name in enumerate(label_names))
	label_to_index

	all_labels = []
	for f in filePaths:
	    if "normal" in f:
	        all_labels.append(1)
	    elif "pneumonia" in f:
	        all_labels.append(0)
	    
	if len(all_labels) != len(filePaths):
	    print("Not working")
	else: 
	    print ("all set")

	return all_labels, label_to_index

def preprocess_image(image):
	image = tf.cast(tf.image.decode_png(image, channels=1),tf.float32)
#   This is a stupid line here but it makes sure that tensorflow knows my image size when I put it in a dataset
	image = tf.image.resize_images(image, [64,64])
	image = image / 255.0  # normalize to [0,1] range
	return image

def load_and_preprocess_image(path):
	image = tf.read_file(path)
	return preprocess_image(image)

def construct_dataset(set):
	filePaths = getFileNames("./"+set+"/")
	all_labels, label_to_index = getLabels(filePaths)

	AUTOTUNE = tf.data.experimental.AUTOTUNE
    
	path_ds = tf.data.Dataset.from_tensor_slices(filePaths)
	
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int64))

	BATCH_SIZE = 32

	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

	image_count = len(filePaths)

	ds = image_label_ds.shuffle(buffer_size=image_count)
	# FUll shuffle when buffer size is image count
	ds = ds.batch(BATCH_SIZE)
	# makes sure we have some data loaded ahead of time to maximize speed
	ds = ds.prefetch(buffer_size=AUTOTUNE)

	return ds




