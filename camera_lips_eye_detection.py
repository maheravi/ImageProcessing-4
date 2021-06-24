import cv2
import random
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')


my_video = cv2.VideoCapture(0)
sticker_eye = cv2.imread('eye.png')
sticker_lip = cv2.imread('lip.png')


while True:

    validation, frame = my_video.read()

    if validation is not True:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, 1.3)

    for f, face in enumerate(faces):
        x, y, w, h = face

        eyes = eye_detector.detectMultiScale(frame_gray[y:y + h, x:x + w])
        smiles = smile_detector.detectMultiScale(frame_gray[y:y + h, x:x + w])

        for i, eye in enumerate(eyes):
            x_e, y_e, w_e, h_e = eye

            resized_sticker = cv2.resize(sticker_eye, (w_e, h_e))
            frame[y+y_e:y+y_e + h_e, x+x_e:x+x_e + w_e] = resized_sticker

        for i, lip in enumerate(smiles):
            x_l, y_l, w_l, h_l = lip

            resized_sticker = cv2.resize(sticker_lip, (w_l, h_l))
            frame[y+y_l:y+y_l + h_l, x+x_l:x+x_l + w_l] = resized_sticker

    cv2.imshow('output', frame)
    cv2.waitKey(10)