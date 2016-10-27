#	coding:utf-8
#
#	CopyRight @makai
#
#	16/10/26
#

from __future__ import division, print_function, absolute_import
import os
import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-type', default='0', help='''0: Show the function of AutoEncoder.
												1: AutoEncoder is used to classify mnist.''')
args = parser.parse_args()

TRAIN_DATA_DIR = 'mnist/mnist_train/'
TEST_DATA_DIR = 'mnist/mnist_test/'

learning_rate = 0.01
p_training_epochs = 20
f_training_epochs = 20
batch_size = 128

n_input = 784
n_h1 = 512
n_h2 = 256
n_output = 10


weights = {
	'ec_h1': tf.Variable(tf.zeros([n_input, n_h1])),
	'ec_h2': tf.Variable(tf.zeros([n_h1, n_h2])),
	'dc_h1': tf.Variable(tf.zeros([n_h2, n_h1])),
	'dc_h2': tf.Variable(tf.zeros([n_h1, n_input])),
	'fc_w': tf.Variable(tf.zeros([n_h2, n_output]))
}

biases = {
	'ec_b1': tf.Variable(tf.zeros([n_h1])),
	'ec_b2': tf.Variable(tf.zeros([n_h2])),
	'dc_b1': tf.Variable(tf.zeros([n_h1])),
	'dc_b2': tf.Variable(tf.zeros([n_input])),
	'fc_b': tf.Variable(tf.zeros([n_output]))
}

def encoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['ec_h1']), biases['ec_b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['ec_h2']), biases['ec_b2']))
	return layer2


def decoder(x):
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['dc_h1']), biases['dc_b1']))
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['dc_h2']), biases['dc_b2']))
	return layer2


def fullconnection(x):
	# out_layer = tf.nn.softmax(tf.add(tf.matmul(x, weights['ful_w']), biases['ful_b']))
	out_layer = tf.nn.softmax(tf.add(tf.matmul(x, weights['fc_w']), biases['fc_b']))
	return out_layer


def getData(dataDir):
	filelist = os.listdir(dataDir)
	data = []
	label = []
	for i, imagename in enumerate(filelist):
		imagename_s = imagename.split('.')
		if imagename_s[-1] == 'jpg':
			im = np.array(Image.open(dataDir + imagename))
			image = np.reshape(im, (n_input))
			data.append(image)
			label.append(int(imagename_s[0]))
	data = np.array(data)
	data = data / 255.
	# label = tf.one_hot(indices=label, depth=10, on_value=1, off_value=0)
	label1 = np.zeros((len(label), 10))
	label1[np.arange(len(label)), label] = 1
	return data, label1


def autoencode():
	# pretrain
	X = tf.placeholder(tf.float32, [None, n_input])
	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)
	p_y_true = X
	p_y_pred = decoder_op
	p_loss = tf.reduce_mean(tf.square(p_y_true - p_y_pred))
	p_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(p_loss)

	# fine-tune
	f_y_true = tf.placeholder('float', [None, 10])
	f_y_pred = fullconnection(encoder_op)
	f_loss = -tf.reduce_sum(f_y_true * tf.log(f_y_pred))	# cross_entropy
	f_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(f_loss)
	correct_pred = tf.equal(tf.argmax(f_y_true, 1), tf.argmax(f_y_pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

	train_data, train_label = getData(TRAIN_DATA_DIR)
	print(train_data.shape)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		total_batch = int(train_data.shape[0] / batch_size)

		# pretrain
		p_train_pred = np.array([])
		for epoch in range(p_training_epochs):
			for i in range(total_batch):
				batch_xs = train_data[i * batch_size : (i + 1) * batch_size]
				_, p_l , train_pred = sess.run([p_optimizer, p_loss, p_y_pred], feed_dict={X: batch_xs})
			print('pretrain---Epoch:' + str(epoch) + ' loss = ' + str(p_l))
		print('\nPretrain Finished!\n')

		test_data, test_label = getData(TEST_DATA_DIR)

		if args.type == '1':
			for epoch in xrange(0, f_training_epochs):
				for i in xrange(0, total_batch):
					# batchsize = 192
					batch_xs = train_data[i * batch_size : (i + 1) * batch_size]
					batch_ys = train_label[i * batch_size : (i + 1) * batch_size]
					_, f_l = sess.run([f_optimizer, f_loss], feed_dict={X: batch_xs, f_y_true: batch_ys})
				y_pred, ac = sess.run([f_y_pred, accuracy], feed_dict={X: batch_xs, f_y_true: batch_ys})
				print('finetune---Epoch:' + str(epoch) + ' loss = ' + str(f_l) + ', training accuracy = ' + str(ac))
			print('\nFinetune Finished!\n')
			test_accuracy = sess.run([accuracy], feed_dict={X: test_data, f_y_true: test_label})
			print('test accuracy:' + str(test_accuracy))
		elif args.type == '0':
			encode_decode = sess.run(p_y_pred, feed_dict={X: test_data[:10]})
	        f, a = plt.subplots(2, 10, figsize=(10, 2))
	        for i in xrange(10):
	        	a[0][i].imshow(np.reshape(test_data[i], (28, 28)), cmap='gray')
	        	a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='gray')
	        f.show()
	        plt.draw()
	        plt.show()

if __name__ == '__main__':
	autoencode()