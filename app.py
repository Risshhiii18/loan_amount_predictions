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
    
    
    
    df = pd.read_csv('https://raw.githubusercontent.com/Risshhiii18/loan_amount_predictions/master/Train.csv')
    df = df.drop(['serial number','credit_amount','credit_history','gurantors','housing_type','foreigner','resident_since'], axis = 1)
    col = []
    def convo(df):
        categorical_data = df.select_dtypes(exclude=[np.number])
        le = LabelEncoder()
        for i in categorical_data.columns:
        #print(dfex[i])
            xe = le.fit_transform(categorical_data[i])
            col.append(xe)
    convo(df)
    categorical_data = df.select_dtypes(exclude=[np.number])
    non_categorical_data = df.select_dtypes(include=[np.number])
    xat = pd.DataFrame(np.transpose(col), columns = categorical_data.columns)
    final_data = pd.concat([xat,non_categorical_data], axis = 1)
    sc = StandardScaler()
    test = sc.fit_transform(final_data)
    
    
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    
    
    df = pd.DataFrame(final_features)
    pred = sc.transform(final_features)
    
    
    
    prediction = model.predict(pred)
    
    output = round(prediction[0])



    return render_template('home.html', prediction_text='You can get some of $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
