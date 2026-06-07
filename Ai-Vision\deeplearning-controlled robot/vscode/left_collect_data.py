import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

left_file = 'left_hand_gestures.csv'

# Clear file + header
with open(left_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'v{i}' for i in range(63)])

# ONLY LEFT HAND GESTURES
gestures = ['Left', 'Right', 'Straight', 'Rotate']
gesture_index = 0
samples = 0
max_samples = 600
is_recording = False

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    if gesture_index >= len(gestures):
        cv2.putText(img, "LEFT HAND DATA DONE!", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    else:
        current_gesture = gestures[gesture_index]

        cv2.putText(img,
                    f"LEFT HAND: {current_gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2)

        if not is_recording:
            cv2.putText(img, "Press SPACE to start", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if key == ord(' '):
                is_recording = True

        else:
            if results.multi_hand_landmarks and results.multi_handedness:

                for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    label = handedness.classification[0].label

                    # ✅ ONLY LEFT HAND
                    if label == "Left":

                        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                        row = [lm.x for lm in hand.landmark] + \
                              [lm.y for lm in hand.landmark] + \
                              [lm.z for lm in hand.landmark]

                        with open(left_file, mode='a', newline='') as f:
                            csv.writer(f).writerow([current_gesture] + row)

                        samples += 1

                        cv2.putText(img,
                                    f"Saving: {samples}/{max_samples}",
                                    (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2)

            if samples >= max_samples:
                is_recording = False
                samples = 0
                gesture_index += 1

    cv2.imshow("LEFT HAND COLLECTOR", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()