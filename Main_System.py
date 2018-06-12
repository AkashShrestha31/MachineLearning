from model import *
from prepare_dataset_for_images import *
im_Size=75
X_train,Y_train,names=prepare_dataset_for_images(100,im_Size,5,"C:/Users/AK/Desktop/problem/Train_Data/")#762
X_test,Y_test,names=prepare_dataset_for_images(25,im_Size,5,"C:/Users/AK/Desktop/problem/Test_Data/")
X_train=X_train/255
X_test=X_test/255
model(X_train,Y_train,X_test,Y_test,im_Size,names)
# model(X,Y)  