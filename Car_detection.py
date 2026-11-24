import cv2, os

# Loadin the .xml file that detects cars
haarfile = r"c:/Users/hjbeb/OneDrive/Desktop/python/Open_Cv/cars.xml"
Cascade = cv2.CascadeClassifier(haarfile)

if Cascade.empty():
    print("Cscade Classifier could not be loaded")
    exit()

# captures each frame of the video
Car_video = cv2.VideoCapture(r"C:/Users/hjbeb/OneDrive/Desktop/python/Open_Cv/Car1.mp4")
cv2.namedWindow("Car detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Car detection", 1000, 1000)

while True:
    # Ret indicates if the frames was captured corrctly. im is the image that the computer returns to us    
    ret, im = Car_video.read()
    if not ret:
        print("Frames where not captured/video was not loaded")
        break
    else:
        height, width = im.shape[:2]
        new_width = 1000
        new_height = 1000
        im = cv2.resize(im, (new_width, new_height))

        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cars_detected = Cascade.detectMultiScale(grey, 1.1, 1)

        # draw rectangle
        for (x, y, w, h) in cars_detected:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 128, 0), 5)

        cv2.imshow("Car detection", im)
        key = cv2.waitKey(10)
        if key == 27:
            break

Car_video.release()
cv2.destroyAllWindows()