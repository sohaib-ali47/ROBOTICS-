import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save(dataset_name, model_name):
    if not os.path.exists(dataset_name):
        print(f"Error: {dataset_name} not found. Please run collect_data.py first.")
        return

    print(f"Loading data from {dataset_name}...")
    df = pd.read_csv(dataset_name)

    X = df.drop('label', axis=1) # 63 columns
    y = df['label']               

    print(f"Training brain for {model_name}...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

    print(f"Success! Brain saved as {model_name}.")

# Automatically train both
if __name__ == "__main__":
    train_and_save('right_hand_gestures.csv', 'right_model.pkl')
    train_and_save('left_hand_gestures.csv', 'left_model.pkl')
    print("All brains are ready for the robot!")