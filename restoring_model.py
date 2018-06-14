import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prepare_dataset_for_images import *

X_test,Y_test,names=prepare_dataset_for_images(25,75,5,"C:/Users/AK/Desktop/problem/Test_Data/")
print(X_test.shape)
def forward_propagation(X,W1,W2):
	Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
	A1=tf.nn.relu(Z1)
	P1=tf.nn.max_pool(A1,ksize =[1,8,8,1],strides=[1,8,8,1],padding="SAME")
	Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
	A2=tf.nn.relu(Z2)
	P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")	
	P = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P, 5, activation_fn=None)
	return Z3
	# Create some variables.
W1 = tf.Variable(tf.zeros([4, 4, 3, 8]))
W2= tf.Variable(tf.zeros([2, 2, 8, 16]))
n_y = Y_test.shape[1]
X=tf.placeholder(tf.float32,[None,75*75])
X = tf.reshape(X, [-1, 75, 75, 3])
Y=tf.placeholder(tf.float32,[None,n_y])
Z3=forward_propagation(X,W1,W2)
init=tf.global_variables_initializer()
# saver = tf.train.Saver()
with tf.Session() as sess:	
	sess.run(init)
	saver= tf.train.import_meta_graph('./model.ckpt.meta')
	saver.restore(sess,tf.train.latest_checkpoint("./model"))
	correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
	  #      # Calculate accuracy on the test set
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(Y_test.shape)
	print("test Accuracy:",accuracy.eval({X:X_test, Y:Y_test,W1:W1.eval(),W2:W2.eval()},session=sess))