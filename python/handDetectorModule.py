import cv2
import mediapipe
import numpy as np
class HandDetector:
    def __init__(self,mode=False,max_hands=2,conf=0.5,tracking_conf=0.5):
        self.drawer=mediapipe.solutions.drawing_utils
        self.mode=mode
        self.hand_detector=mediapipe.solutions.hands.Hands(static_image_mode=mode,min_detection_confidence=conf,min_tracking_confidence=tracking_conf,max_num_hands=max_hands)
        self.max_hands=max_hands

    def DetectHand(self,frame,draw=True):
            self.hand=self.hand_detector.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            mask=np.zeros(frame.shape)
            
            if draw:
                if self.hand.multi_hand_landmarks:
                    for h in self.hand.multi_hand_landmarks:  
                        self.drawer.draw_landmarks(mask,h,mediapipe.solutions.hands.HAND_CONNECTIONS)
        
                    
            return frame,mask
    def handPostion(self,frame,hand_num=0,draw=True):
        
        landmarks=[]
        if self.hand.multi_hand_landmarks:
                myHand=self.hand.multi_hand_landmarks[hand_num]
                # for h in self.hand.multi_hand_landmarks[hand_num]:
                for ids,lm in enumerate(myHand.landmark):
                            h,w,c=frame.shape
                            cx,cy= int(lm.x*w),int(lm.y*h)
                            landmarks.append([ids,cx,cy,lm.z])

        return landmarks


