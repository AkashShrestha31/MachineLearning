import tensorflow as tf
from prepare_dataset_for_images import *
import numpy as np
from scipy import ndimage
import scipy.misc
X_test,Y_test,names=prepare_dataset_for_images(25,75,5,"C:/Users/AK/Desktop/problem/Test_Data/")
sess=tf.Session() 
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = "images"
output_key = "scores"

export_path =  "./savemodel" 
meta_graph_def = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],export_path)
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)

# y_out = sess.run(y, {x: X_test})
# print(y_out)
# correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(Y_test,1))
#        # Calculate accuracy on the test set
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 		# print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
# print(sess.run(correct_prediction))

path="C:/Users/AK/Desktop/problem/Train_Data/Sirsha/sirsha14.jpg"
image=np.array(ndimage.imread(path,flatten=False))
if image.shape[2]>3:
	image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
print(image.shape)

image=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
image=np.reshape(image,(image.shape[0],75,75,3))
y_out = sess.run(y, {x: image})
prediction=tf.argmax(y_out,1)
# value=sess.run(prediction,{x:image})
value=np.squeeze(sess.run(prediction))
print(value)

print(image.shape)


