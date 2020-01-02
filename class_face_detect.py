# import this file into training_data.py

# import whatever needed

import cv2
import numpy as np
import os

# put 'haarcascade_frontalface_alt.xml' in this python file folder
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# object for face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# function to detect faces
def dect_face(a):
    l = ()
    faces = face_detect.detectMultiScale(a,1.3,5)
    for (x,y,w,h) in faces:
        a = a[y:y+h,x:x+w]
        l = l+(x,y,x+w,y+h)
    return a, l

# function to recognize face from a pic
def find_img(test,names):
    img = cv2.imread(test)
    im_g = cv2.imread(test,0)
    face,rect = dect_face(im_g)
    lable = face_recognizer.predict(face)
    draw_rect(img,rect,names[lable[0]])
    return img

# function to draw rect and display name if any face recognized
def draw_rect(img,rect,name):
    if len(rect) != 0:
        x1,y1,x2,y2 = rect
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(img,name,(x1,y1),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),1)

# function to find faces from webcam 
def find_cam(test,names):
    img = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    face,rect = dect_face(img)
    lable = face_recognizer.predict(face)
    draw_rect(img,rect,names[lable[0]])
    return img

# function to prepare date for the training 
#should contain folders with name 'numbers' and the overall data path = data_path
def prep_data(data_path):
    faces = []
    labels = []
    os.chdir(data_path)
    for dir_name in os.listdir():
        data_path_new = data_path
        data_path_new = data_path_new + '\\' + dir_name
        print(data_path_new)
        os.chdir(data_path_new)
        for image in os.listdir():
            im = cv2.imread(image,0)
            face,rect = dect_face(im)
            faces.append(face)
            labels.append(int(dir_name))
    return faces,labels

# function to train face recognizer
def train(faces,lables):
    face_recognizer.train(faces, np.array((lables)))


# these functions will called from training_data.py













