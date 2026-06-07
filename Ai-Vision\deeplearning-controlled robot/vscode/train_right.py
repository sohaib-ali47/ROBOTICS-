import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load your right-hand data
data_file = 'right_hand_gestures.csv'

if not os.path.exists(data_file):
    print(f"CRITICAL ERROR: {data_file} not found! Check your filename.")
    exit()

print(f"Loading {data_file} (expecting 250 samples per gesture)...")
df = pd.read_csv(data_file)

# 2. Separate Labels and Features
# Ensure no extra columns are in the CSV
X = df.drop('label', axis=1) 
y = df['label']              

# 3. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Right Brain
print("Training Right Hand brain...")
model = RandomForestClassifier(n_estimators=100) # Increased estimators for 250 samples
model.fit(X_train, y_train)

# 5. Check Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Right Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Right Brain
with open('right_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Success! Right brain saved as 'right_model.pkl'.")