import os
import cv2
import time
import  numpy as np

def dect_face(a):
    l = ()
    faces = face_detect.detectMultiScale(a,1.3,5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(a,(x,y),(x+w,y+h),(255,0,0),2)
        a = a[y:y+h,x:x+w]
        l = l+(x,y,x+w,y+h)
    return a, l

def prep_data():
    faces = []
    labels = []
    data_path = 'C:\\Users\MANOJI M\Desktop\\face recognition\\train'
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


def find(test):
    img = cv2.imread(test)
    im_g = cv2.imread(test,0)
    face,rect = dect_face(im_g)
    lable = face_recognizer.predict(face)
    draw_rect(img,rect,names[lable[0]])
    return img

def draw_rect(img,rect,name):
    x1,y1,x2,y2 = rect
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.putText(img,name,(x1,y1),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),1)



face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
names = ['','Mechanic','Rajni']

faces,lables = prep_data()
print(len(faces))
print(len(lables))

face_recognizer.train(faces,np.array((lables)))

os.chdir('C:\\Users\MANOJI M\Desktop\\face recognition')
#print(names[find('2.jpg')[0]])
im = find('1.jpg')
cv2.imshow('dec', im)
cv2.waitKey(0)
cv2.destroyAllWindows()








