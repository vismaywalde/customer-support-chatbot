import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocess import preprocess_text, train_word2vec, get_sentence_vector
import numpy as np
import pickle

data = pd.read_csv('data/customer_support_queries.csv')
data['tokens'] = data['query'].apply(preprocess_text)
w2v_model = train_word2vec(data['tokens'].tolist())
X = np.array([get_sentence_vector(tokens, w2v_model) for tokens in data['tokens']])
le = LabelEncoder()
y = le.fit_transform(data['intent'])
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=8)
model.save('models/chatbot_model.h5')
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
