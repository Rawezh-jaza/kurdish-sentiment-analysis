from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
import joblib
import tempfile
import os
import pandas as pd 
import re
import nltk
from kurdish import ku 

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
with open('static/model/svm_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)
    
def preprocessText(text):
    if pd.isnull(text):
        return ""
    cleaned_text = re.sub(r"<.*?>", "", text)
    cleaned_text = re.sub(r'@[a-zA-Z0-9_]+\s?[a-zA-Z0-9_]+', "", cleaned_text)
    cleaned_text = re.sub(r"http\S+|www\S+|https\S+", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text, flags=re.UNICODE)
    cleaned_text = re.sub(r"[0-9]", "", cleaned_text)
    words = nltk.word_tokenize(cleaned_text)
    cleaned_text = " ".join(words)
    cleaned_text = ku.Hemwar().ali_k_to_uni(cleaned_text)
    return cleaned_text

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
    
    preprocessed_text = preprocessText(text)
    
    predicted_label = model.predict([preprocessed_text])[0]

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
        predicted_label = model.predict([sentence])[0]

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
