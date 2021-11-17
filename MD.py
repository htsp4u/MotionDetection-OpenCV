import cv2
import numpy as np
 
# Capturing video, creating windows and changing location
video = cv2.VideoCapture(0)
cv2.namedWindow('Regular',cv2.WINDOW_NORMAL)
cv2.moveWindow('Regular', 0,0)
cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)
cv2.moveWindow('Gray', 0,330)
cv2.namedWindow('Difference',cv2.WINDOW_NORMAL)
cv2.moveWindow('Difference', 1000,0)
cv2.namedWindow('Threshold',cv2.WINDOW_NORMAL)
cv2.moveWindow('Threshold', 1000,330)
cv2.namedWindow('Motion Detect',cv2.WINDOW_NORMAL)

while True: 
    _, frame = video.read() 
    _, frame2 = video.read()
    _, frame3 = video.read()

    # Show regular image
    cv2.imshow('Regular', frame)

    # Create multiple frames so that images can be
    # compared constantly
    frame2 = frame3
    frame3 = frame
    
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Converting gray scale image to GaussianBlur  
    # so that change can be found easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
      
    # Difference between previous frame  
    # and current frame(which are GaussianBlur and gray filtered) 
    diff_frame = cv2.absdiff(gray2, gray) 
  
    # If change in between previous frame and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
  
    # Finding contour of moving object 
    (cnts, _) = cv2.findContours(thresh_frame.copy(),  
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle arround the moving object 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
  
    # Displaying image in gray_scale 
    cv2.imshow("Gray", gray) 
  
    # Displaying the difference in current frame to 
    # the previous frame 
    cv2.imshow("Difference", diff_frame) 
  
    # Displaying the black and white image in which if 
    # intencity difference greater than 30 it will appear white 
    cv2.imshow("Threshold", thresh_frame) 
  
    # Displaying color frame with contour of motion of object 
    cv2.imshow("Motion Detect", frame)

    # Resize all windows to declutter
    cv2.resizeWindow('Regular', (300,300))
    cv2.resizeWindow('Gray', (300,300))
    cv2.resizeWindow('Difference', (300,300))
    cv2.resizeWindow('Threshold', (300,300))
    cv2.resizeWindow('Motion Detect', (600,600))
    
    # if q entered whole process will stop
    key = cv2.waitKey(1) 
    if key == ord('q'): 
        break

# Turn off camera
video.release() 
  
# Destroying all the windows 
cv2.destroyAllWindows() 
