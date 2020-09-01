import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib


app = Flask(__name__)
model = joblib.load('vott.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    
    
    prediction = model.predict(final_features)
    
    output = round(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
