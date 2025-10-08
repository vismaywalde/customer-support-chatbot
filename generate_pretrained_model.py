import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a tiny dummy dataset
X_dummy = np.random.rand(6, 100)  # 6 samples, 100 features
y_dummy = np.array([0, 1, 2, 0, 2, 1])  # 3 classes: refund, order_status, complaint

# Label encoder mapping
label_mapping = {0: "refund", 1: "order_status", 2: "complaint"}

# Build a small ANN model
model = Sequential([
    Dense(32, input_dim=100, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model briefly
model.fit(X_dummy, y_dummy, epochs=5, verbose=1)

# Save model and label encoder
os.makedirs("AI-Customer-Support-Chatbot/models", exist_ok=True)
model.save("AI-Customer-Support-Chatbot/models/chatbot_model.h5")

with open("AI-Customer-Support-Chatbot/models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_mapping, f)

print("Pre-trained model and label encoder created successfully!")
