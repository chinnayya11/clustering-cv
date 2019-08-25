import cv2
import numpy 
# #read an image in color. Use 0 for greyscale
x=cv2.imread("pikachu.png",1)
# # import pdb; pdb.set_trace()
cv2.imshow("image",x)
cv2.waitKey(0)
cv2.destroyAllWindows()
# #write to a file
cv2.imwrite("pikachu1.jpg",x)

# #video processing for the same operations
cap=cv2.VideoCapture(0)#pass file name
while(True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#draw on image
img=cv2.imread("pikachu1.jpg",1)
img = cv2.rectangle(img,(10,0),(30,20),(0,255,0),3)
img = cv2.circle(img,(50,30), 20, (0,0,255), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,100), font, 1,(0,255,255),2,cv2.LINE_AA)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#retrieve outline manually
import cv2
import numpy as np

img=cv2.imread("pikachu1.jpg",1)
new_img=numpy.zeros([225,225,1])

# # import pdb; pdb.set_trace()
h,w=img.shape[:2]
for i in range(225):
     for j in range(225):
         m=min(img.item(i,j,0),img.item(i,j,1),img.item(i,j,2))
         if m<50:
             m=0
         new_img[i,j,0]=m
         # img[i,j,0]=numpy.ones([225,225])*0
cv2.imshow("Modified",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()