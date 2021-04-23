from imutils import face_utils
import time
import dlib
from aspect_ratio import aspect_ratio
import cv2
from playsound import playsound
import numpy as np


counter = 0
face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

(left_eye_Start, left_eye_End) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(right_eye_Start, right_eye_End) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_webcam = cv2.VideoCapture(0)
time.sleep(2)

while(True):

    ret, frame = video_webcam.read()                   #read frame
    frame = cv2.flip(frame,1)                           #flip frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #convert frame to grayscale
    face = face_detector(gray, 0)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for face in face:
        shape = face_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_Eye = shape[left_eye_Start:left_eye_End]
        right_Eye = shape[right_eye_Start:right_eye_End]

        l_ar = aspect_ratio(left_Eye)
        r_ar = aspect_ratio(right_Eye)
        eyeAspectRatio = (l_ar + r_ar) / 2

        l_Hull = cv2.convexHull(left_Eye)
        r_Hull = cv2.convexHull(right_Eye)
        cv2.drawContours(frame, [l_Hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [r_Hull], -1, (0, 255, 0), 1)

        if(eyeAspectRatio < 0.3):
            counter += 1
            if counter >= 50:
                playsound('assets/alert.wav')
                cv2.putText(frame, "DROWSY", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,50,255), 2)
        else:
            counter = 0

    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        breakvideo_capture.release()
cv2.destroyAllWindows()
