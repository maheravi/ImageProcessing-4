import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_video = cv2.VideoCapture('Obama.mp4')

while True:

    validation, frame = my_video.read()

    if validation is not True:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame, 1.3)

    for i, face in enumerate(faces):
        x, y, w, h = face

        width, height = (16, 16)
        temp = cv2.resize(frame[y:y + h, x:x + w], (width, height), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f'kalle ha/kalle{i}.png', output)
        frame[y:y + h, x:x + w] = output

    cv2.imshow('output', frame)
    cv2.waitKey(10)