# Hand Detection Project for Game Automation
import cv2
import mediapipe as mp
import time
from directkeys import RIGHT_ARROW_SCANCODE, LEFT_ARROW_SCANCODE
from directkeys import PressKey, ReleaseKey  

# Key definitions
brake_key = LEFT_ARROW_SCANCODE
gas_key = RIGHT_ARROW_SCANCODE

time.sleep(2.0)  # Delay before starting
current_keys_pressed = set()

# MediaPipe setup
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

tipIds = [4, 8, 12, 16, 20]

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        key_pressed = None
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)  # Mirror the camera
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lmList = []
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        fingers = []
        if lmList:
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            if total_fingers == 5:
                cv2.putText(frame, "GAS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                PressKey(gas_key)
                key_pressed = gas_key

            elif total_fingers == 0:
                cv2.putText(frame, "BRAKE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                PressKey(brake_key)
                key_pressed = brake_key

        # Release previous key if different
        if key_pressed:
            for key in current_keys_pressed:
                if key != key_pressed:
                    ReleaseKey(key)
            current_keys_pressed = {key_pressed}
        else:
            for key in current_keys_pressed:
                ReleaseKey(key)
            current_keys_pressed = set()

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
