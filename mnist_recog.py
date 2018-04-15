"""
@author: Yosiyoshi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys, os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy
from PIL import Image
from matplotlib import pylab as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def deepnn(x):
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
	return y_conv, keep_prob

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 2000, 0.1, staircase = True)
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step = global_step)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

#mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_test = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in range(300):
	batch_xs, batch_ys =mnist.train.next_batch(100) #mnist.train.next_batch(100)#
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
	if i%100 == 0:
		train_accuracy = 0
		for j in range(100):
			batch_xs, batch_ys =mnist_test.train.next_batch(100)
			train_accuracy = train_accuracy + accuracy.eval(session = sess, feed_dict={x:batch_xs,  y_: batch_ys, keep_prob:1})
		lr = sess.run(learning_rate)
		print ("step %d, lr=%g ,training accuracy %g"%(i, lr, train_accuracy/100))
        
  #Import as glayscale images
name = 'input_number.jpg'
img = Image.open(name).convert('L')
plt.imshow(img)
  #transform them into 28*28 size
img.thumbnail((28, 28))
  #float
img = numpy.array(img, dtype=numpy.float32)
img = 1 - numpy.array(img / 255)
img = img.reshape(1, 784)
p = sess.run(y_conv, feed_dict={x:img, y_: [[0.0] * 10], keep_prob: 0.5})[0]
  #prediction
print(name+"is recognized as:")
print(numpy.argmax(p))