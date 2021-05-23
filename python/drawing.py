from handDetectorModule import *
import cv2

cap = cv2.VideoCapture(0)
hands=HandDetector(conf=0.75,tracking_conf=0.75,mode=True)
temp=None
cap.set(3, 1080)
cap.set(4, 720)
xo,yo=None,None

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if frame is None:
        continue
    frame,mask=hands.DetectHand(frame)
    
    index=hands.handPostion(frame,0)
    if temp is None:
        temp=np.zeros(frame.shape)
    if len(index)>0:
        
        r=hands.isup(frame,[0])
        
        if r[1] and r[2] is True:
            mode=modes['select']
            
        else:
            if r[1] is True or r[2] is False:
                [x1,y1]=[int((a+b)/2) for a,b in zip(index[8][0:2],index[12][0:2])]
                if xo is None:
                    xo,yo=x1,y1
                    continue
                else:
                    xo , yo=None,None
                cv2.line(temp,(x1,y1),(xo,yo),(255,0,255),3,)
                xo,yo=x1,y1
                
    mask=mask+temp
    frame=cv2.addWeighted(frame,0.6,mask.astype(np.uint8),0.4,0)
    cv2.imshow('frame',frame)
    # cv2.imshow('text',temp)

    # cv2.imshow('mask',mask)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()