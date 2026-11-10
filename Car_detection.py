import cv2

#Loadin the .xml file that detects cars
haarfile = cv2.CascadeClassifier("C:/Users/hjbeb/OneDrive/Desktop/python/Open_Cv/cars.xml")

#captures each frame of the video
Car_video = cv2.VideoCapture("C:/Users/hjbeb/OneDrive/Desktop/python/Open_Cv/Car1.mp4")
cv2.namedWindow("Car detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Car detection", 1000,1000)

while True:
    #Ret indicates if the frames was captured corrctly. im is the image that the computer returns to us    
    ret,im = Car_video.read()
    if not ret:
            print("Frames where not captured/video was not loaded")
            break
    else:
        height,width = im.shape[:2]
        new_width =1000
        new_height = 1000
        cv2.resize(im,(new_width,new_height))