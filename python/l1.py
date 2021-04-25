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
    if len(index)>0:
        [x1,y1],[x2,y2]=index[4][0:2],index[8][0:2]
        cv2.circle(mask,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(mask,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(mask,(x1,y1),(x2,y2),(255,0,0),3)
        d=dist([x1,y1],[x2,y2])
        r=hands.isup(frame,[0])
        print(r)
        # cv2.putText(mask,)


    # cv2.imshow('Image',frame)
    cv2.imshow('mask',mask)
    
    # masked=cv2.bitwise_and(frame,frame,mask=mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()