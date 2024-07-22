from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)

model = joblib.load('kazakh_spam_classifier_rf_model.pkl')
vectorizer = joblib.load('kazakh_tfidf_vectorizer.pkl')


def predict_spam(email_text):
    email_features = vectorizer.transform([email_text])
    prediction = model.predict(email_features)
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json.get('email_text')
    result = predict_spam(email_text)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
