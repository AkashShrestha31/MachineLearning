import tensorflow as tf
def create_placeholders(n_H0,n_W0,n_C0,n_y,im_Size):
	X=tf.placeholder(tf.float32,[None,im_Size*im_Size])
	x_image = tf.reshape(X, [-1, im_Size, im_Size, 3])
	Y=tf.placeholder(tf.float32,[None,n_y])
	return x_image,Y
	