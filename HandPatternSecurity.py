import cv2
import mediapipe as mp
import numpy as np

class HandprintSecuritySystem:
    def __init__(self):
        self.detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.saved_handprints = []  # List to store saved handprints

    def capture_handprint(self, img):
        landmarks, _ = self.detect_landmarks(img)

        # Check if hand is detected
        if landmarks:
            # Append the landmarks to the list of saved handprints
            self.saved_handprints.append(landmarks)
            print("Handprint captured successfully!")
            cv2.imshow("Captured Handprint", img)
            cv2.waitKey(0)

    def authenticate_handprint(self, img):
        landmarks, _ = self.detect_landmarks(img)

        # Check if hand is detected
        if landmarks:
            # Compare the landmarks with saved handprints for authentication
            for saved_handprint in self.saved_handprints:
                if self.compare_landmarks(landmarks, saved_handprint):
                    print("Handprint authenticated successfully!")
                    return True
            print("Authentication failed!")
        else:
            print("No hand detected!")

        return False

    def detect_landmarks(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    # Store the normalized coordinates of landmarks
                    landmarks.append((lm.x, lm.y))
                return landmarks, hand_landmarks

        return None, None

    def compare_landmarks(self, landmarks1, landmarks2, threshold=0.1):
        # Compare landmarks using Euclidean distance
        distances = np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1)
        mean_distance = np.mean(distances)
        return mean_distance < threshold

def main():
    security_system = HandprintSecuritySystem()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Capture a handprint
        key = cv2.waitKey(1)
        if key == ord('c'):
            security_system.capture_handprint(frame)
        
        # Authenticate handprint
        if security_system.authenticate_handprint(frame):
            # Implement your security measures here
            # For demonstration, let's just print a message
            print("Access granted!")

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
