import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
# Track 2 hands so the script can see which is which
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

right_file = 'right_hand_gestures.csv'
left_file = 'left_hand_gestures.csv'

# Clear files and add headers
for file_name in [right_file, left_file]:
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'v{i}' for i in range(63)])

# Your specific routing tasks
tasks = [
    {'hand': 'Right', 'gesture': 'Forward', 'file': right_file},
    {'hand': 'Right', 'gesture': 'Backward', 'file': right_file},
    {'hand': 'Right', 'gesture': 'Stop', 'file': right_file},
    {'hand': 'Left', 'gesture': 'Left', 'file': left_file},
    {'hand': 'Left', 'gesture': 'Right', 'file': left_file},
    {'hand': 'Left', 'gesture': 'Straight', 'file': left_file},
    {'hand': 'Left', 'gesture': 'Rotate', 'file': left_file}
]

task_index = 0
samples = 0
max_samples = 600
is_recording = False

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    key = cv2.waitKey(1) & 0xFF
    
    if task_index >= len(tasks):
        cv2.putText(img, "ALL DONE! Press 'q'", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        current = tasks[task_index]
        cv2.putText(img, f"TASK: Use {current['hand']} hand for {current['gesture']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if not is_recording:
            cv2.putText(img, "Press SPACE to start", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if key == ord(' '): is_recording = True
        else:
            if results.multi_hand_landmarks and results.multi_handedness:
                # Find the hand that matches the required side
                for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label # 'Right' or 'Left'
                    if label == current['hand']:
                        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                        row = [lm.x for lm in hand.landmark] + [lm.y for lm in hand.landmark] + [lm.z for lm in hand.landmark]
                        with open(current['file'], mode='a', newline='') as f:
                            csv.writer(f).writerow([current['gesture']] + row)
                        samples += 1
                        cv2.putText(img, f"Saving: {samples}/{max_samples}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if samples >= max_samples:
                is_recording = False
                samples = 0
                task_index += 1

    cv2.imshow("Dual-Channel Collector", img)
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()