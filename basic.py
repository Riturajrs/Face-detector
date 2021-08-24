import cv2,time

face_detect = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")
vid = cv2.VideoCapture(0)

while True:
    check, frame = vid.read()
    # frame = cv2.resize(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)))
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Camera",gray_img)
    eye = eye_detect.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    face = face_detect.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in face:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    for x,y,w,h in eye:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Camera",frame)
    key = cv2.waitKey(1)
    if( key == ord('q') ):
        break
vid.release()
cv2.destroyAllWindows
