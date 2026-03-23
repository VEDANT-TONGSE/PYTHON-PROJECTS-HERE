import mediapipe as mp
import cv2
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
mp_hands = mp.solutions.hands
hands=mp_hands.Hands()
mp_draw=mp.solutions.drawing_utils
strt_time=0
end_time=0
l,d=[],[]
while True:
    l=[]
    success,frame=cap.read()
    img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_lms,mp_hands.HAND_CONNECTIONS)
            for i,ii in enumerate(hand_lms.landmark):
                if i!=4 and i!=8:
                    continue
                h,w,c=frame.shape
                cx,cy=int(ii.x*w),int(ii.y*h)
                l.append([cx,cy])
                # cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
                # cv2.putText(frame,str(i),(cx,cy),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
            try:
                d.append(((l[1][1]-l[0][1])**2+(l[1][0]-l[0][0])**2)**0.5)
            except:
                pass
    try:
        volume.SetMasterVolumeLevelScalar(d[-1]/max(d), None) 
    except:
        pass
    strt_time=time.time()
    frame_rate=1/(strt_time-end_time)
    cv2.putText(frame,f'FPS:{int(frame_rate)}',(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
    end_time=strt_time
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
