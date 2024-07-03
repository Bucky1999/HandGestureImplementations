import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import HandTracking as ht
import autopy
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDesktopWidget
from PyQt5.QtGui import QIcon


class GestureApplication(QWidget):
    def __init__(self):
        super().__init__()

        # Set window title and icon
        self.setWindowTitle('Gesture Application')
        self.setWindowIcon(QIcon('icon2.ico'))

        # Set window size and position
        self.resize(500, 500)
        self.center()

        # Create buttons
        self.button1 = QPushButton('Virtual Mouse')
        self.button2 = QPushButton('Volume Mouse')

        # Add buttons to layout
        layout = QVBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)

        # Set layout
        self.setLayout(layout)

        # Connect button signals to slots
        self.button1.clicked.connect(self.virtual_mouse_clicked)
        self.button2.clicked.connect(self.volume_mouse_clicked)

    def center(self):
        # Get screen geometry
        screen_geometry = QDesktopWidget().screenGeometry()

        # Calculate center position
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2

        # Set window position
        self.move(x, y)

    def virtual_mouse_clicked(self):
        # Define functionality for virtual mouse button click
        ### Variables Declaration
        pTime = 0               # Used to calculate frame rate
        width = 640             # Width of Camera
        height = 480            # Height of Camera
        frameR = 100            # Frame Rate
        smoothening = 8         # Smoothening Factor
        prev_x, prev_y = 0, 0   # Previous coordinates
        curr_x, curr_y = 0, 0   # Current coordinates

        cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
        cap.set(3, width)           # Adjusting size
        cap.set(4, height)

        detector = ht.handDetector(maxHands=1)                  # Detecting one hand at max
        screen_width, screen_height = autopy.screen.size()      # Getting the screen size
        while True:
            success, img = cap.read()
            img = detector.findHands(img)                       # Finding the hand
            lmlist, bbox = detector.findPosition(img)           # Getting position of hand

            if len(lmlist)!=0:
                x1, y1 = lmlist[8][1:]
                x2, y2 = lmlist[12][1:]

                fingers = detector.fingersUp()      # Checking if fingers are upwards
                cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)   # Creating boundary box
                if fingers[1] == 1 and fingers[2] == 0:     # If fore finger is up and middle finger is down
                    x3 = np.interp(x1, (frameR,width-frameR), (0,screen_width))
                    y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))

                    curr_x = prev_x + (x3 - prev_x)/smoothening
                    curr_y = prev_y + (y3 - prev_y) / smoothening

                    autopy.mouse.move(screen_width - curr_x, curr_y)    # Moving the cursor
                    cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                if fingers[1] == 1 and fingers[2] == 1:     # If fore finger & middle finger both are up
                    length, img, lineInfo = detector.findDistance(8, 12, img)

                    if length < 40:     # If both fingers are really close to each other
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()    # Perform Click

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", img)

            # Check for spacebar key press to close the code
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

    def volume_mouse_clicked(self):
        # Define functionality for volume mouse button click
        # solution APIs
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        # Volume Control Library Usage 
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volRange = volume.GetVolumeRange()
        minVol , maxVol , volBar, volPer= volRange[0] , volRange[1], 400, 0

        # Webcam Setup
        wCam, hCam = 1980, 1080
        cam = cv2.VideoCapture(0)
        cam.set(3,wCam)
        cam.set(4,hCam)

        # Mediapipe Hand Landmark Model
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

          while cam.isOpened():
            success, image = cam.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    )

            # multi_hand_landmarks method for Finding postion of Hand landmarks      
            lmList = []
            if results.multi_hand_landmarks:
              myHand = results.multi_hand_landmarks[0]
              for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])          

            # Assigning variables for Thumb and Index finger position
            if len(lmList) != 0:
              x1, y1 = lmList[4][1], lmList[4][2]
              x2, y2 = lmList[8][1], lmList[8][2]

              # Marking Thumb and Index finger
              cv2.circle(image, (x1,y1),15,(255,255,255))  
              cv2.circle(image, (x2,y2),15,(255,255,255))   
              cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
              length = math.hypot(x2-x1,y2-y1)
              if length < 50:
                cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)

              vol = np.interp(length, [50, 220], [minVol, maxVol])
              volume.SetMasterVolumeLevel(vol, None)
              volBar = np.interp(length, [50, 220], [400, 150])
              volPer = np.interp(length, [50, 220], [0, 100])

              # Volume Bar
              cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
              cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
              cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 0), 3)

            cv2.imshow('handDetector', image) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        cam.release()
    

if __name__ == '__main__':
    # Create PyQt application
    app = QApplication(sys.argv)

    # Create instance of the GUI application
    gesture_app = GestureApplication()

    # Show the GUI
    gesture_app.show()

    # Execute the application
    sys.exit(app.exec_())
