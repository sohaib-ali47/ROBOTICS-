import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Training LEFT hand brain...")
df = pd.read_csv('left_hand_gestures.csv')

X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open('left_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Success! Left hand model saved.")