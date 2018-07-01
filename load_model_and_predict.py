import tensorflow as tf
from prepare_dataset_for_images import *
import numpy as np
from scipy import ndimage
import scipy.misc
import numpy as np
import shutil
import os
import numpy as np
import time
# X_test,Y_test,names=prepare_dataset_for_images(25,75,5,"C:/Users/AK/Desktop/problem/Test_Data/")
from taking_input import *
X=np.zeros((5,5,5))
names=["Dagina","Abinash","Pawan","Gokul","Sirsha"]
haar_face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
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
def predict(X,index,middle):
	folder="./faces/"
	files=os.listdir(folder)
	for i in range(len(files)):
		path=folder+files[i]
		image=np.array(ndimage.imread(path,flatten=False))
		if image.shape[2]>3:
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
		# print(image.shape)
		image=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
		image=np.reshape(image,(image.shape[0],75,75,3))
		y_out = sess.run(y, {x: image})
		correct_prediction=tf.argmax(y_out,1)
		value=sess.run(correct_prediction)
		value=np.squeeze(value)
		print(value)
		if value==0:
			print(names[0])
			# dagina[0,inxex]=1
			X[0,middle,index]=1
		if value==1:
			print(names[1])
			# abinash[0,index]=1
			X[1,middle,index]=1
		if value==2:
			print(names[2])
			# pawan[0,index]=1
			X[2,middle,index]=1
		if value==3:
			print(names[3])
			# gokul[0,index]=1
			X[3,middle,index]=1
		if value==4:
			print(names[4])
			# sirsha[0,index]=1
			X[4,middle,index]=1
		# print(abinash)
	print(X[0,0,:])
	print(X[1,0,:])
	print(X[2,0,:])   7126698640D
	print(X[3,0,:])
	print(X[4,0,:])

def extract_face(path):
	try:
		shutil.rmtree('./faces')
		os.makedirs('./faces')
	except:
		os.makedirs('./faces')
	image=cv2.imread(path)
	image2=image.copy()
	faces = haar_face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5);  #1.3
	j=0
	#print the number of faces found 
	print('Faces found: ', len(faces))
	#go over list of faces and draw them as rectangles on original colored 
	for (x, y, w, h) in faces:     
	         cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 4)
	         try:
	         	check=image2[y-10:y+10+h,x-10:x+w+10].shape
	         	if check[0]>150:
	         		plt.imsave("./faces/"+str(j)+".jpg",cv2.cvtColor(image2[y-10:y+10+h,x-10:x+w+10],cv2.COLOR_BGR2RGB))
	         		j=j+1
	         except:
	         	print("error message")
	         	continue
def load_model_and_predict(middle):
	for i in range(5):
		path="./frames/capture"+str(i)+".jpg"
		print("times: ",i)    	   	
		extract_face(path)
		predict(X,i,middle)
		time.sleep(5)
	print("The final result is ")
for i in range(5):
	for j in range(5):
		taking_input(j)
	load_model_and_predict(i)
print(X)







































# np.random.seed(0)
# folder="C:/Users/AK/Desktop/Tensorflow/dataset/captureim/"
# files=os.listdir(folder)
# shuffle=list(np.random.permutation(len(files)))

# print(files)
# for i in range(5):
# 	# print(files[i])
# 	path=folder+files[shuffle[i]]
# 	print(path)
# 	extract_face(path,files[shuffle[i]])
	# image=np.array(ndimage.imread(path,flatten=False))
	# if image.shape[2]>3:
	# 	image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	# print(image.shape)

	# image=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
	# image=np.reshape(image,(image.shape[0],75,75,3))