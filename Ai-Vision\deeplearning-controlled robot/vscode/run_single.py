import cv2
import mediapipe as mp
import pickle
import pandas as pd

# CHANGE THIS LINE to test 'right_model.pkl' or 'left_model.pkl'
model_to_load = 'right_model.pkl' 

with open(model_to_load, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
        # Extract 63 coordinates
        row = [lm.x for lm in hand.landmark] + \
      [lm.y for lm in hand.landmark] + \
      [lm.z for lm in hand.landmark]
            
        # Predict
        X = pd.DataFrame([row], columns=[f'v{i}' for i in range(63)])
        gesture = model.predict(X)[0]
        
        cv2.putText(img, f"TESTING {model_to_load}: {gesture}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Isolation Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()