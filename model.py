from compute_cost import *
from create_placeholder import *
from forward_propogation import *
import matplotlib.pyplot as plt
from initialize_parameters import *
from random_mini_batches import *
import scipy.misc
import cv2
from scipy import ndimage
haar_face_cascade=cv2.CascadeClassifier("C:/Users/AK/Desktop/problem/haarcascade_frontalface_alt.xml")
def model(X_train, Y_train,X_test,Y_test,im_Size,names,learning_rate=0.009,
          num_epochs=10, minibatch_size=64, print_cost=True):
	#create placeholder
	tf.set_random_seed(1) 
	(m,n_H0,n_W0,n_C0)=X_train.shape
	n_y = Y_train.shape[1]
	costs=[]
	X,Y=create_placeholders(n_H0,n_W0,n_C0,n_y,im_Size)
	# np.random.seed(0)
	seed=3 
	#initailize parameter
	parameters=initialize_parameters()
	#forward propogatation
	Z3=forward_propagation(X,parameters)
	#compute the cost
	cost=compute_cost(Z3,Y)
	#create an optimizer
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
		plt.show()
		#Saving Model
		print("Saving model")
		saver=tf.train.Saver()
		saver.save(sess,"C:/Users/AK/Desktop/problem/Saved_Model/model")
		correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
       # Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
		parameters=sess.run(parameters)
		sess.close()
		#Loading model
	print("Retriving saved model and comparing weights")
	sess=tf.Session()
	saver= tf.train.import_meta_graph('C:/Users/AK/Desktop/problem/Saved_Model/model.meta')
	saver.restore(sess,tf.train.latest_checkpoint("C:/Users/AK/Desktop/problem/Saved_Model/./"))
	if sess.run("W1:0").all()==parameters["W1"].all():
		print("Value W1 is same before and after model saving and retrival ")
	else:
		print("Not same result")
	if sess.run("W2:0").all()==parameters["W2"].all():
		print("Value W2 is same before and after model saving and retrival")
	else:
		print("Not same result")
	#RUN RESTORING_MODEL TO TEST THE MODEL WHETHER THE SAVED MODEL GIVES SAME RESULT DURING THE TRAING THE MODEL
		

