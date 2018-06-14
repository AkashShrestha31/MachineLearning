from compute_cost import *
from create_placeholder import *
from forward_propogation import *
import matplotlib.pyplot as plt
from initialize_parameters import *
from random_mini_batches import *
import scipy.misc
import cv2
import sys,os
from scipy import ndimage
haar_face_cascade=cv2.CascadeClassifier("C:/Users/AK/Desktop/problem/haarcascade_frontalface_alt.xml")
def model(X_train, Y_train,X_test,Y_test,im_Size,names,learning_rate=0.009,
          num_epochs=20, minibatch_size=64, print_cost=True):
	#create placeholder
	# tf.set_random_seed(1) 
	(m,n_H0,n_W0,n_C0)=X_train.shape
	n_y = Y_train.shape[1]
	costs=[]
	X,Y=create_placeholders(n_H0,n_W0,n_C0,n_y,im_Size)
	# np.random.seed(0)
	seed=3 
	#initailize parameter
	parameters=initialize_parameters()
	#forward propogatation
	W1=parameters["W1"]
	W2=parameters["W2"]
	
	Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
	A1=tf.nn.relu(Z1)
	P1=tf.nn.max_pool(A1,ksize =[1,8,8,1],strides=[1,8,8,1],padding="SAME")
	Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
	A2=tf.nn.relu(Z2)
	P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")	
	P = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P, 5, activation_fn=None)
	#compute the cost
	cost=compute_cost(Z3,Y)
	#create an optimizer
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	saver=tf.train.Saver()
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost=0
			num_minibatches = int(m / minibatch_size)
			batches=random_mini_batches(X_train,Y_train,64,seed)
			seed=seed+1
			for batch in batches:
				minibatch_X,minibatch_Y=batch
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				minibatch_cost += temp_cost / num_minibatches
            # Print the cost every epoch
			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
				correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
			# if print_cost == True and epoch % 1 == 0:
			costs.append(minibatch_cost)
        # plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		
		correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
       # Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
		print(correct_prediction.eval({X: X_test, Y: Y_test}))
		parameters=sess.run(parameters)
		#Loading model
		export_path_base = "./savemodel"

		print('Exporting trained model to', export_path_base)
		builder = tf.saved_model.builder.SavedModelBuilder(export_path_base)

		tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
		tensor_info_y = tf.saved_model.utils.build_tensor_info(Z3)

		prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={'images': tensor_info_x},outputs={'scores': tensor_info_y},method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
		builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature})
		builder.save()
		print('Done exporting!')
 
	#RUN RESTORING_MODEL TO TEST THE MODEL WHETHER THE SAVED MODEL GIVES SAME RESULT DURING THE TRAING THE MODEL
		

