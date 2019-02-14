import tensorflow as tf
import numpy as np
from .drawer import crop
from .union import combine

class SimpleNet(object):
	labels = list()
	C = int()
	model = str()

	def __init__(self, yolo):
		self.model = yolo.model
		self.S = yolo.S
		self.labels = yolo.labels
		self.C = len(self.labels)

		base = int(np.ceil(pow(self.C, 1./3)))

		self.inp = tf.placeholder(tf.float32, [None, 448, 448, 3], name = 'input')
		self.drop = tf.placeholder(tf.float32, name = 'dropout')
		
		now = self.inp
		for i in range(yolo.layer_number):
			l = yolo.layers[i]
			if l.type == 'CONVOLUTIONAL':
				if l.pad < 0:
					size = np.int(now.get_shape()[1])
					expect = -(l.pad + 1) * l.stride # there you go bietche 
					expect += l.size - size
					padding = [int (expect / 2), int(expect - expect / 2)]
					if padding[0] < 0: padding[0] = 0
					if padding[1] < 0: padding[1] = 0
				else:
					padding = [int(l.pad), int(l.pad)]
				l.pad = 'VALID'
				now = tf.pad(now, [[0, 0], padding, padding, [0, 0]])
				
				b = tf.Variable(l.biases)
				w = tf.Variable(l.weights)
				now = tf.nn.conv2d(now, w,
					strides=[1, l.stride, l.stride, 1],
					padding=l.pad)
				now = tf.nn.bias_add(now, b)
				now = tf.maximum(0.1 * now, now)			
			elif l.type == 'MAXPOOL':
				l.pad = 'VALID'
				now = tf.nn.max_pool(now, 
					padding = l.pad,
					ksize = [1,l.size,l.size,1], 
					strides = [1,l.stride,l.stride,1])			
			elif l.type == 'FLATTEN':
				now = tf.transpose(now, [0,3,1,2])
				now = tf.reshape(now, 
					[-1, int(np.prod(now.get_shape()[1:]))])			
			elif l.type == 'CONNECTED':
				name = str()
				if i == yolo.layer_number - 1: name = 'output'
				else: name = 'conn'
				
				b = tf.Variable(l.biases)
				w = tf.Variable(l.weights)
				now = tf.nn.xw_plus_b(now, w, b, name = name)
			elif l.type == 'LEAKY':
				now = tf.maximum(0.1 * now, now)
			elif l.type == 'DROPOUT':
				now = tf.nn.dropout(now, keep_prob = self.drop)
		self.out = now

	def setup_meta_ops(self, gpu):
		if gpu > 0: 
			
    		#config.gpu_options.visible_device_list = "0"
    		
			percentage = min(gpu, 1.)
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=percentage,visible_device_list = "0")
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = True,
				log_device_placement = False,
				gpu_options = gpu_options))
		else:
			gpu_options = tf.GPUOptions(visible_device_list = "0")
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = False,
				log_device_placement = False,
				gpu_options = gpu_options))

		self.sess.run(tf.initialize_all_variables())

	def mul(self, box, w, h):
		box['x'] *= w
		box['w'] *= w
		box['y'] *= h
		box['h'] *= h
		return box

	def predict(self, img, threshold, merge):
		img, w, h= crop(img)

		prehold = threshold
		if merge: prehold /= 3

		feed_dict = {
			self.inp : np.concatenate([img, img[:,:,::-1,:]], 0), 
			self.drop : 1.0
		}

		out = self.sess.run([self.out], feed_dict)

		boxes = list()
		predictions = out[0]
		flip = False
		for prediction in predictions:
			SS = self.S ** 2
			prob_size = SS * 1
			conf_size = SS * 2

			probs = prediction[0 : prob_size]
			confs = prediction[prob_size : (prob_size + conf_size)]
			cords = prediction[(prob_size + conf_size) : ]

			probs = probs.reshape([SS, 1])
			confs = confs.reshape([SS, 2])
			cords = cords.reshape([SS, 2, 4])

			for grid in range(SS):
				for b in range(2):
					box = {'w':0, 'h':0, 'x':0, 'y':0, 'p':0}
					box['x'] = (cords[grid, b, 0] + grid %  self.S) / self.S
					box['x'] = 1 - box['x'] if flip else box['x']
					box['y'] = (cords[grid, b, 1] + grid // self.S) / self.S
					box['w'] =  cords[grid, b, 2] ** 2
					box['h'] =  cords[grid, b, 3] ** 2
					box['p'] = confs[grid, b] * probs[grid, 0]
					if box['p'] > prehold:
						boxes.append(box)

			if flip: break
			flip = True

		if merge: boxes = combine(boxes, w, h, threshold)
		return [ self.mul(box, w, h) for box in boxes]
