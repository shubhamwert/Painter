from handDetectorModule import *
import cv2
cap = cv2.VideoCapture(0)
hands=HandDetector(conf=0.6,tracking_conf=0.6,mode=True)


while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if frame is None:
        continue
    frame,mask=hands.DetectHand(frame)
    index=hands.handPostion(frame,0)
    thumb=hands.handPostion(frame,0)
    # cv2.imshow('Image',frame)
    cv2.imshow('mask',mask)
    
    # masked=cv2.bitwise_and(frame,frame,mask=mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()