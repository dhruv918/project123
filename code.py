import cv2 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from PIL import Image 
import PIL.ImageOps 
import os, ssl, time

X = np.load('image (1).npz')['arr_0']
y = pd.read_csv("label.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
x_train1=x_train/255.0
x_test1=x_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train1, y_train)
y_prd=y_train.predict(x_test1)
accuracy=accuracy_score(y_test,y_prd)
cap=cv2.VideoCapture(0)
while (True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape 
        upper_left = (int(width / 2 - 56), int(height / 2 - 56)) 
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56)) 
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert('L')
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixelfilter=20
        minimumpixel=np.percentile(image_bw_resize_inverted,pixelfilter)
        maximumpixel=np.max(image_bw_resize_inverted,pixelfilter)
        image_bw_resize_inverted_scale=np.clip(image_bw_resize_inverted-minimumpixel,0,255)
        image_bw_resize_inverted_scale=np.asarray(image_bw_resize_inverted_scale)/maximumpixel
        test_sample=np.array(image_bw_resize_inverted_scale).reashape(1,784)
        test_prd=clf.predict(test_sample)
        cv2.imshow('frame')
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
    cap.release()
    cv2.destroyAllWindows()







        




