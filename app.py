from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welocom():
    return 'Welcom all'


@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The prediction value is " + str(prediction)


@app.route('/predict_dict',methods = ['POST'])
def predict_dict():
    request_data = request.json
    data = request_data['data']
    #prediction = classifier.predict([input_data])
    return jsonify({'Message':"The prediction value is " + data})

@app.route('/predict_file', methods = ['POST'])
def predict_file():
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return 'The predicted values for the CSV is '+ str(list(prediction))


if __name__ == "__main__":
    app.run()