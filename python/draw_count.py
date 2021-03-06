from handDetectorModule import *
import cv2
import tensorflow.keras.models as Models
from matplotlib import pyplot as plt

def Croped(img):
    a=np.where(img==255)
    # print(len(a))
    x1=sorted(a[0])[0]
    x2=sorted(a[0])[-1]
    y1=sorted(a[1])[0]
    y2=sorted(a[1])[-1]
    return cv2.resize(img[x1-10:x2+10,y1-10:y2+10],(28,28))
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
k=False
prediction=0
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
        if r[4] is True & r[3] is True & k is True:
                if all(temp.shape) > 0 :
                    print(k)

                    temp=Croped(temp)
                   
                    temp=temp/255.0
                    temp=temp[:,:,0]
                    n=Prediction_Number.predict(temp.reshape([1,28,28,1]))
                    # print('|'*10,'\n',n[0].argmax())
                    prediction=10*prediction+n[0].argmax()
                    temp=None
                    k=False
                    continue
        if r[1] and r[2] is True:
            xo,yo=None,None

            
        else:
            if r[1] is True and r[2] is False:
                [x1,y1]=index[8][0:2]
                k=True
                if xo is None:
                    xo,yo=x1,y1
                    continue
                cv2.line(temp,(x1,y1),(xo,yo),(255,0,255),9,10)
                xo,yo=x1,y1
    # mask=mask+temp
    frame=cv2.addWeighted(frame,0.6,temp.astype(np.uint8),0.4,0)
    # temp=mask
    cv2.putText(frame,str(prediction),org=(50,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=3,fontScale=2,color=(32,255,2))
    
    cv2.imshow('frame',frame)
    # cv2.imshow('text',temp)

    # cv2.imshow('mask',mask)
    c=temp
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()