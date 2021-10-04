import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands = mpHands.Hands(False)   #creating object
#already given deafult values in hands.phy
#if it has a good tracking confidence then it will keep tracking or else will do detection again
mpDraw = mp.solutions.drawing_utils  #will use mediapipe function

#now we will ad frame rate or fps
pTime = 0 #previous time
cTime = 0 #current time


#handdetectionmodel (already implemented)


while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #sending rgb image
    results=hands.process(imgRGB)  #it will process the frame and will give us the result
    #will extract the obecjts and extract muultiple hands to check will apply for loop to extract multiple hands
    #print(results.multi_hand_landmarks) #to check values when we put the hands to get the landmarks of points

    if results.multi_hand_landmarks:  #
        for handLms in results.multi_hand_landmarks:#here we will have each hand and extract info of each hand
             for id,lm in enumerate(handLms.landmark):#relate to finger landmark

                 h,w,c = img.shape #get widht and hieght
                 cx, cy = int(lm.x*w),int(lm.y*h)     #integer decimal places
                 print(id,cx,cy) #wiritng id and printing id no cx and cy posotion in order
                 if id == 0:
                     cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)



             mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)#drawing connections using mpHands.HAND_CONNECTIONS

             #will use mediapipe function mpdraw defined above to check 21 landmarsk of infgers
#till here it can detect all the landmarks of the finers and hands because of google mediapipe

    cTime = time.time()
    fps = 1 / (cTime - pTime)   #to display fps
    pTime = cTime                #previous time will become current time


    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255), 3)
#in above line
    #


    cv2.imshow("Image" , img)
    cv2.waitKey(1)