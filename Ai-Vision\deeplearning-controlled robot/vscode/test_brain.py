import pandas as pd
import pickle
import numpy as np

# Load the Right Brain
with open('right_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a "fake" hand row of zeros to see what it predicts
fake_hand = np.zeros((1, 63))
prediction = model.predict(fake_hand)

print(f"If the robot shows all zeros, it predicts: {prediction[0]}")

# Now load your training data and check what the model thinks about it
df = pd.read_csv('right_hand_gestures.csv')
sample = df.drop('label', axis=1).iloc[0:5] # Look at first 5 real samples
predictions = model.predict(sample)
print(f"When looking at real training data, it predicts: {predictions}")