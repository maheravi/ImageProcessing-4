import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_video = cv2.VideoCapture('Obama.mp4')

while True:
    validation, frame = my_video.read()

    if validation is not True:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rows, cols = frame_gray.shape

    Q1 = frame_gray[0:rows//2, 0:cols//2]
    Q2 = frame_gray[rows // 2:rows, 0:cols // 2]
    Q3 = frame_gray[0:rows // 2, cols // 2:cols]
    Q4 = frame_gray[rows // 2:rows, cols // 2:cols]

    Q1 = cv2.flip(Q1, 1)
    Q2 = cv2.flip(Q2, 0)
    Q3 = cv2.flip(Q3, 1)
    Q4 = cv2.flip(Q4, 0)

    frame_gray[0:rows // 2, 0:cols // 2] = Q1
    frame_gray[rows // 2:rows, 0:cols // 2] = Q2
    frame_gray[0:rows // 2, cols // 2:cols] = Q3
    frame_gray[rows // 2:rows, cols // 2:cols] = Q4

    cv2.imshow('output', frame_gray)
    cv2.waitKey(10)