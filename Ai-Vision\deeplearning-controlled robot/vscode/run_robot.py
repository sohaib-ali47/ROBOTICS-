import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np

# for communication with Arduino
import serial
import time

arduino = serial.Serial('COM7', 9600) # change COM port if needed
time.sleep(2)

# ---------------- SERIAL FUNCTIONS ----------------

def send_command(cmd):
    arduino.write(cmd.encode())


last_cmd = ""
last_time = 0

def safe_send(cmd):
    global last_cmd, last_time

    if cmd != last_cmd or time.time() - last_time > 0.2:
        send_command(cmd)
        last_cmd = cmd
        last_time = time.time()


# ---------------- LOAD MODELS ----------------

with open('right_model.pkl', 'rb') as f:
    right_model = pickle.load(f)

with open('left_model.pkl', 'rb') as f:
    left_model = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- MAIN LOOP ----------------

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Temporary state variables to hold current frame's signals
    # Defaulting to safe values if a hand is missing from the frame
    current_right_signal = "Stop"
    current_left_signal = "Straight"

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # -------- FEATURE EXTRACTION (63 values) --------
            row = [lm.x for lm in hand.landmark] + \
                  [lm.y for lm in hand.landmark] + \
                  [lm.z for lm in hand.landmark]

            X = pd.DataFrame([row], columns=[f'v{i}' for i in range(63)])

            hand_label = handedness.classification[0].label

            # -------- PREDICTION --------
            if hand_label == "Right":
                gesture = right_model.predict(X)[0]
                current_right_signal = gesture  
            else:
                gesture = left_model.predict(X)[0]
                current_left_signal = gesture   

            # -------- DEBUG --------
            print(f"{hand_label} Hand: {gesture}")

            color = (0, 255, 0) if hand_label == "Right" else (0, 0, 255)
            cv2.putText(img,
                        f"{hand_label}: {gesture}",
                        (10, 50 if hand_label == "Left" else 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

    # ---------------- COMBINED SIGNAL LOGIC ----------------
    def mix_signals(right, left):
        if right == "Stop":
            # If right is stop but left wants a sharp point-rotation
            if left == "Left":     return 'L'  # Sharp Rotate Left (Tank Turn)
            if left == "Right":    return 'R'  # Sharp Rotate Right (Tank Turn)
            return 'S'                         # Hard Stop
            
        if right == "Forward":
            if left == "Left":     return 'Q'  # Slight Turn Left while moving Forward
            if left == "Right":    return 'E'  # Slight Turn Right while moving Forward
            return 'F'                         # Forward Straight

        if right == "Backward":
            if left == "Left":     return 'Z'  # Slight Turn Left while moving Backward
            if left == "Right":    return 'C'  # Slight Turn Right while moving Backward
            return 'B'                         # Backward Straight
            
        return 'S'

    # Mix and safely transmit the consolidated command frame
    final_mixed_cmd = mix_signals(current_right_signal, current_left_signal)
    safe_send(final_mixed_cmd)

    cv2.imshow("Robot Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()