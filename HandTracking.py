import cv2
import numpy as np
import time

from src.hand_tracker import HandTracker

#Initialize constants and load paths
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
POINT_COLOR = (0, 0, 0)
CONNECTION_COLOR = (76, 36, 133)
THICKNESS = 5

def handtracking():
    #Initialize webcam capture
    cv2.namedWindow(WINDOW)
    capture = cv2.VideoCapture(0)
    record = False

    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False

    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-

    #All connections made to detect hand from the diagram above
    connections = [
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17), (0, 5),
        (0, 17), (0, 9), (0, 13), (0, 17), (0, 2), (0, 1), (1, 5),
        (0, 6), (0, 10), (0, 14), (0, 18), (1, 9), (1, 13),
        (1, 17), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)
    ]

    #Palm connections
    palm = [(0, 9), (0, 13), (0, 17), 
            (0, 1), (1, 5),(1, 9), (1, 13),
            (1, 17), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]


    detector = HandTracker(
        PALM_MODEL_PATH,
        LANDMARK_MODEL_PATH,
        ANCHORS_PATH,
        box_shift=0.2,
        box_enlarge=1.3
    )

    #Initialize Video Writer to save video clip for model to predict on
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter('testvideo.mp4', fourcc, 5, size) 


    while hasFrame:

                
        #Convert images to different color channels for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #Masking image to isolate the hand
        hsv_color1 = np.asarray([0, 0, 255])   
        hsv_color2 = np.asarray([76, 36, 133])

        mask = cv2.inRange(image_hsv, hsv_color1, hsv_color2)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if record == False:
            cv2.putText(mask,"Press Space To Start Recording or Q to quit", (50,50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
          
        #Detect points on hand
        points, _ = detector(image)
        if points is not None:
            for point in points:
                x, y = point

            #Join points using connections
            for connection in connections:
                
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                
                if connection in palm:
                    cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS*5)
                    
                else:
                    cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS*3)


        cv2.namedWindow("Sign2Text", cv2.WINDOW_NORMAL)
        

        horizontal_concat = np.concatenate((mask, frame), axis=1)

        cv2.resizeWindow('Sign2Text', 1000,1000)

        cv2.imshow('Sign2Text', horizontal_concat)



        hasFrame, frame = capture.read()
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key % 256 == 32:
            record = True
            cv2.rectangle(mask, (0,0), (1240,100), (0,0,0), -1)
        if record:
            out.write(mask)

    capture.release()
    out.release()
    
cv2.destroyAllWindows()
