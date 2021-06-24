import cv2
import random
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_video = cv2.VideoCapture(0)
sticker1 = cv2.imread('sticker1.png')
sticker2 = cv2.imread('sticker2.png')
sticker3 = cv2.imread('sticker3.png')
sticker4 = cv2.imread('sticker4.png')
sticker5 = cv2.imread('sticker5.png')

while True:

    validation, frame = my_video.read()

    if validation is not True:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, 1.3)

    for i, face in enumerate(faces):
        x, y, w, h = face

        sticker = random.choice([sticker1, sticker2, sticker3, sticker4, sticker5])
        resized_sticker = cv2.resize(sticker, (w, h))

        frame_face = frame[y:y + h, x:x + w]
        cv2.imwrite(f'kalle ha/kalle{i}.png', frame_face)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 8)
        frame[y:y + h, x:x + w] = resized_sticker

    cv2.imshow('output', frame)
    cv2.waitKey(10)