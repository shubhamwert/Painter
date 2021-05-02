from handDetectorModule import *
import cv2
import tensorflow.keras.models as Models

cap = cv2.VideoCapture(0)
hands=HandDetector(conf=0.75,tracking_conf=0.75,mode=True)
temp=None

cap.set(3, 1080)
cap.set(4, 720)
Prediction_Number=Models.load_model('model')
xo,yo=None,None
modes={
        'select':-1,
        'red':0,
        'blue':1,
        'green':2


}
mode=modes['select']
colors={'red':(255,0,0),'blue':(0,255,0),'green':(0,0,255)}
color_selected='red'
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if frame is None:
        continue
    frame,mask=hands.DetectHand(frame)
    if temp is None:
        temp=np.zeros(frame.shape)
    index=hands.handPostion(frame,0)
    if len(index)>0:
        
        r=hands.isup(frame,[0])
        if r[4] is True and r[3] is True:
                
                    temp.resize([1,28,28])
                    temp.reshape([28,28,1])
                    n=Prediction_Number.predict(temp)

                    print('|'*10,'\n',n[0].argmax())
                    temp=None
        if r[1] and r[2] is True:
            xo,yo=None,None

            
        else:
            if r[1] is True and r[2] is False:
                [x1,y1]=index[8][0:2]
                if xo is None:
                    xo,yo=x1,y1
                    continue
                cv2.line(temp,(x1,y1),(xo,yo),(255,0,255),3,)
                xo,yo=x1,y1
    mask=mask+temp
    frame=cv2.addWeighted(frame,0.6,mask.astype(np.uint8),0.4,0)
    
    cv2.imshow('frame',mask)
    # cv2.imshow('text',temp)

    # cv2.imshow('mask',mask)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()