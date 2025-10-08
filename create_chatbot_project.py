import os
import zipfile

project_name = "AI-Customer-Support-Chatbot"
folders = [
    f"{project_name}/data",
    f"{project_name}/src",
    f"{project_name}/models",
    f"{project_name}/notebooks"
]

files = {
    f"{project_name}/data/customer_support_queries.csv": 
    "query,intent\n"
    "I want to return my order,refund\n"
    "Where is my package?,order_status\n"
    "My order arrived damaged,complaint\n"
    "Can I cancel my subscription?,refund\n"
    "My product is defective,complaint\n"
    "How long does shipping take?,order_status\n",

    f"{project_name}/src/preprocess.py":
    "import nltk\n"
    "from nltk.tokenize import word_tokenize\n"
    "from nltk.stem import WordNetLemmatizer\n"
    "import numpy as np\n"
    "from gensim.models import Word2Vec\n\n"
    "nltk.download('punkt')\n"
    "nltk.download('wordnet')\n\n"
    "lemmatizer = WordNetLemmatizer()\n\n"
    "def preprocess_text(text):\n"
    "    tokens = word_tokenize(text.lower())\n"
    "    tokens = [lemmatizer.lemmatize(t) for t in tokens]\n"
    "    return tokens\n\n"
    "def train_word2vec(sentences, vector_size=100, window=5, min_count=1):\n"
    "    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)\n"
    "    return model\n\n"
    "def get_sentence_vector(sentence_tokens, w2v_model):\n"
    "    vectors = [w2v_model.wv[token] for token in sentence_tokens if token in w2v_model.wv]\n"
    "    if len(vectors) == 0:\n"
    "        return np.zeros(w2v_model.vector_size)\n"
    "    return np.mean(vectors, axis=0)\n",

    f"{project_name}/src/train_model.py":
    "import pandas as pd\n"
    "from sklearn.preprocessing import LabelEncoder\n"
    "from tensorflow.keras.models import Sequential\n"
    "from tensorflow.keras.layers import Dense\n"
    "from preprocess import preprocess_text, train_word2vec, get_sentence_vector\n"
    "import numpy as np\n"
    "import pickle\n\n"
    "data = pd.read_csv('data/customer_support_queries.csv')\n"
    "data['tokens'] = data['query'].apply(preprocess_text)\n"
    "w2v_model = train_word2vec(data['tokens'].tolist())\n"
    "X = np.array([get_sentence_vector(tokens, w2v_model) for tokens in data['tokens']])\n"
    "le = LabelEncoder()\n"
    "y = le.fit_transform(data['intent'])\n"
    "model = Sequential([\n"
    "    Dense(128, input_dim=X.shape[1], activation='relu'),\n"
    "    Dense(64, activation='relu'),\n"
    "    Dense(len(le.classes_), activation='softmax')\n"
    "])\n"
    "model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n"
    "model.fit(X, y, epochs=50, batch_size=8)\n"
    "model.save('models/chatbot_model.h5')\n"
    "with open('models/label_encoder.pkl', 'wb') as f:\n"
    "    pickle.dump(le, f)\n",

    f"{project_name}/src/chatbot.py":
    "from tensorflow.keras.models import load_model\n"
    "import pickle\n"
    "from preprocess import preprocess_text, get_sentence_vector, train_word2vec\n"
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "data = pd.read_csv('data/customer_support_queries.csv')\n"
    "data['tokens'] = data['query'].apply(preprocess_text)\n"
    "w2v_model = train_word2vec(data['tokens'].tolist())\n"
    "model = load_model('models/chatbot_model.h5')\n"
    "with open('models/label_encoder.pkl', 'rb') as f:\n"
    "    le = pickle.load(f)\n\n"
    "def predict_intent(query):\n"
    "    tokens = preprocess_text(query)\n"
    "    vector = get_sentence_vector(tokens, w2v_model)\n"
    "    vector = vector.reshape(1, -1)\n"
    "    pred = model.predict(vector)\n"
    "    intent = le.inverse_transform([np.argmax(pred)])\n"
    "    return intent[0]\n\n"
    "print('Chatbot: Hello! How can I help you today? (type \"quit\" to exit)')\n"
    "while True:\n"
    "    query = input('You: ')\n"
    "    if query.lower() == 'quit':\n"
    "        print('Chatbot: Goodbye!')\n"
    "        break\n"
    "    intent = predict_intent(query)\n"
    "    print(f'Chatbot (predicted intent): {intent}')\n",

    f"{project_name}/requirements.txt":
    "pandas\nnumpy\nnltk\ngensim\ntensorflow\nscikit-learn\n",

    f"{project_name}/README.md":
    "# AI-Powered Customer Support Chatbot\n\n"
    "This is an AI-powered customer support chatbot that uses **NLP preprocessing** and an **ANN** model to classify customer queries into intents like refund, order status, complaints, etc.\n\n"
    "## How to Run\n"
    "1. Install dependencies:\n"
    "```\npip install -r requirements.txt\n```\n"
    "2. Train the model:\n"
    "```\npython src/train_model.py\n```\n"
    "3. Run the chatbot:\n"
    "```\npython src/chatbot.py\n```\n",

    f"{project_name}/.gitignore":
    "*.pyc\n__pycache__/\nmodels/\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)

# Create zip
zip_filename = f"{project_name}.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, filenames in os.walk(project_name):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            zipf.write(filepath, os.path.relpath(filepath, project_name))

print(f"{zip_filename} created successfully!")
