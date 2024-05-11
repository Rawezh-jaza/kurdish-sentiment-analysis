from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
import pickle
import tempfile
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained sentiment analysis model (classifier)
with open('static/model/nb_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('static/model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']    
    if not text:
        return jsonify({'variable': 'تکایە ڕستەیەک بنووسە'})
    
    # Vectorize the text data using the pre-trained TF-IDF vectorizer
    text_data = vectorizer.transform([text])
    
    # Predict using the loaded classifier
    predicted_label = model.predict(text_data)[0]

    if predicted_label == 'positive':
        label = 'ئەم ڕستە ئەرێنییە✔️'
    elif predicted_label == 'neutral':
        label = 'ئەم ڕستە بێلایەنە🤷'
    elif predicted_label == 'negative':
        label = 'ئەم ڕستەیە نەرێنییە❌'

    return jsonify({'variable': label})

@app.route('/downloads/<filename>')
def download_file(filename):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found"

@app.route('/predict_file', methods=['POST'])
def predict_file():
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'result': 'تکایە فایلێکی ئیگزڵ هەڵبژێرە (xlsx, xls)'})

    _, temp_path = tempfile.mkstemp()
    file.save(temp_path)

    df = pd.read_excel(temp_path)
    text_column = request.form['column']

    if text_column not in df.columns:
        return jsonify({'result': 'ناوی کۆڵۆمەکە هەڵەیە'})

    analyzed_sentences = []
    for index, row in df.iterrows():
        sentence = row[text_column]
        text_data = vectorizer.transform([sentence])
        predicted_label = model.predict(text_data)[0]

        if predicted_label == 'positive':
            label = 'Positive'
        elif predicted_label == 'neutral':
            label = 'Neutral'
        else:
            label = 'Negative'

        analyzed_sentences.append([sentence, label])

    df['Sentiment'] = [label for _, label in analyzed_sentences]

    result_file_path = os.path.join(tempfile.gettempdir(), 'analyzed_file.xlsx')
    df.to_excel(result_file_path, index=False)

    return jsonify({
        'table': df.to_html(classes='table table-bordered', index=False),
        'download_link': '/downloads/analyzed_file.xlsx'
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
