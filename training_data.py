# import class_face_detect to import its functions
import cv2
import class_face_detect
import os

# names of ur persons
names = ['','Mechanic','Rajni']

# class_face_detect functions are made as objects to be used
detect_face = class_face_detect.dect_face
find_img = class_face_detect.find_img
find_cam = class_face_detect.find_cam
draw_rect = class_face_detect.draw_rect
prep_data = class_face_detect.prep_data
train = class_face_detect.train

# data_path
a = 'C:\\Users\MANOJI M\Desktop\\face recognition\\train'
# data preparation
faces,lables = prep_data(a)
print(len(faces))
print(len(lables))
train(faces,lables)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = find_cam(frame,names)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



