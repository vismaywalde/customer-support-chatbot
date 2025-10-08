from tensorflow.keras.models import load_model
import pickle
from preprocess import preprocess_text, get_sentence_vector, train_word2vec
import pandas as pd
import numpy as np

data = pd.read_csv('data/customer_support_queries.csv')
data['tokens'] = data['query'].apply(preprocess_text)
w2v_model = train_word2vec(data['tokens'].tolist())
model = load_model('models/chatbot_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def predict_intent(query):
    tokens = preprocess_text(query)
    vector = get_sentence_vector(tokens, w2v_model)
    vector = vector.reshape(1, -1)
    pred = model.predict(vector)
    intent = le.inverse_transform([np.argmax(pred)])
    return intent[0]

print('Chatbot: Hello! How can I help you today? (type "quit" to exit)')
while True:
    query = input('You: ')
    if query.lower() == 'quit':
        print('Chatbot: Goodbye!')
        break
    intent = predict_intent(query)
    print(f'Chatbot (predicted intent): {intent}')
