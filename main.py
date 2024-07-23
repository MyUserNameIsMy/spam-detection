from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)

model_kz = joblib.load('kazakh_spam_classifier_rf_model.pkl')
vectorizer_kz = joblib.load('kazakh_tfidf_vectorizer.pkl')

model_ru = joblib.load('russian_toxic_classifier_rf_model.pkl')
vectorizer_ru = joblib.load('russian_tfidf_vectorizer.pkl')

model_en = joblib.load('english_spam_classifier_rf_model.pkl')
vectorizer_en = joblib.load('english_tfidf_vectorizer.pkl')


def predict_spam_kz(email_text):
    email_features = vectorizer_kz.transform([email_text])
    prediction = model_kz.predict(email_features)
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"


def predict_toxic_ru(email_text):
    email_features = vectorizer_ru.transform([email_text])
    prediction = model_ru.predict(email_features)
    if prediction[0] == 1:
        return "Toxic"
    else:
        return "Not Toxic"


def predict_spam_en(email_text):
    email_features = vectorizer_en.transform([email_text])
    prediction = model_en.predict(email_features)
    if prediction[0] == 1:
        return "Spam"
    else:
        return "Not Spam"


@app.route('/predict_spam_kz', methods=['POST'])
def predict_kz():
    email_text = request.json.get('email_text')
    result = predict_spam_kz(email_text)
    return jsonify({"result": result})


@app.route('/predict_toxic_ru', methods=['POST'])
def predict_ru():
    email_text = request.json.get('email_text')
    result = predict_toxic_ru(email_text)
    return jsonify({"result": result})


@app.route('/predict_spam_en', methods=['POST'])
def predict_en():
    email_text = request.json.get('email_text')
    result = predict_spam_en(email_text)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
