from flask import Flask, request, jsonify
import pickle

# Load the trained model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Spam Detection API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid input. Please provide a message.'}), 400
    
    # Extract the message
    message = data['message']
    
    # Vectorize the message
    message_vec = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_vec)
    probability = model.predict_proba(message_vec).max()
    
    # Map prediction result
    result = {
        'message': message,
        'prediction': prediction[0],
        'confidence': f"{probability * 100:.2f}%"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
