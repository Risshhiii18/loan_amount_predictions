import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import joblib


app = Flask(__name__)
model = joblib.load('vot1.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    
    
    df = pd.DataFrame(final_features)
    sc = StandardScaler()
    test = sc.fit_transform(df)
    
    prediction = model.predict(test)
    
    output = round(prediction[0], 2)



    return render_template('home.html', prediction_text='You can get some of $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
