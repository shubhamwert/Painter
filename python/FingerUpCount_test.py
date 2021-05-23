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
        
        r=hands.isup(frame,[0])
        c=sum(r.values())-1
        cv2.putText(mask,str(c),org=(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=3,fontScale=2,color=(32,255,2))


    cv2.imshow('mask',mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()