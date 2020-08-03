import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
#import pickle
import joblib
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('svr_model.pkl')
dum_cols = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index - Copy.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = list(str(x) for x in request.form.values())
    example = [
    {'job title': features[0], 'location':features[1], 'size':features[2], 'type of ownership':features[3],'python':features[4], 'rstudio':features[5], 'sql' : features[6], 'aws':features[7], 'pytorch':features[8], 'sas':features[9], 'excel': features[10]} 
     ]
    example = pd.DataFrame(example)
    example = pd.get_dummies(example)
    example= example.reindex(columns = dum_cols, fill_value = 0)
    pred = model.predict(example)

    output = int(round(pred[0]))

    return render_template('index - Copy.html', prediction_text=' CA$ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
